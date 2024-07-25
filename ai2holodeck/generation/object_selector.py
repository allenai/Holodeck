import ast
import copy
import json
import multiprocessing
import random
import re
import traceback
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from colorama import Fore
from langchain import PromptTemplate, OpenAI
from shapely import Polygon

import ai2holodeck.generation.prompts as prompts
from ai2holodeck.generation.floor_objects import DFS_Solver_Floor
from ai2holodeck.generation.objaverse_retriever import ObjathorRetriever
from ai2holodeck.generation.types import (
    object_plan_from_dict,
    ObjectPlanDict,
    FloorOrWallObjectDict,
)
from ai2holodeck.generation.utils import (
    get_bbox_dims,
    get_annotations,
    get_bbox_dims_vec,
)
from ai2holodeck.generation.wall_objects import DFS_Solver_Wall

EXPECTED_OBJECT_ATTRIBUTES = [
    "description",
    "location",
    "size",
    "quantity",
    "variance_type",
    "objects_on_top",
]


class ObjectSelector:
    def __init__(self, object_retriever: ObjathorRetriever, llm: OpenAI):
        # object retriever
        self.object_retriever = object_retriever
        self.database = object_retriever.database

        # language model and prompt templates
        self.llm = llm
        self.object_selection_template_1 = prompts.object_selection_prompt_new_1
        self.object_selection_template_2 = PromptTemplate(
            input_variables=[
                "object_selection_prompt_new_1",
                "object_selection_1",
                "room",
            ],
            template=prompts.object_selection_prompt_new_2,
        )

        # hyperparameters
        self.floor_capacity_ratio = 0.4
        self.wall_capacity_ratio = 0.5
        self.object_size_tolerance = 0.8
        self.similarity_threshold_floor = 31  # need to be tuned
        self.similarity_threshold_wall = 31  # need to be tuned
        self.thin_threshold = 3
        self.used_assets = []
        self.consider_size = True
        self.size_buffer = 10

        self.random_selection = False
        self.reuse_selection = False
        self.multiprocessing = True

    def select_objects(self, scene, additional_requirements="N/A"):
        rooms_types = [room["roomType"] for room in scene["rooms"]]
        room2area = {
            room["roomType"]: self.get_room_area(room) for room in scene["rooms"]
        }
        room2size = {
            room["roomType"]: self.get_room_size(room, scene["wall_height"])
            for room in scene["rooms"]
        }
        room2perimeter = {
            room["roomType"]: self.get_room_perimeter(room) for room in scene["rooms"]
        }
        room2vertices = {
            room["roomType"]: [(x * 100, y * 100) for (x, y) in room["vertices"]]
            for room in scene["rooms"]
        }

        room2floor_capacity = {
            room_type: [room_area * self.floor_capacity_ratio, 0]
            for room_type, room_area in room2area.items()
        }
        room2floor_capacity = self.update_floor_capacity(room2floor_capacity, scene)
        room2wall_capacity = {
            room_type: [room_perimeter * self.wall_capacity_ratio, 0]
            for room_type, room_perimeter in room2perimeter.items()
        }
        selected_objects = {
            room["roomType"]: {"floor": [], "wall": []} for room in scene["rooms"]
        }

        if "object_selection_plan" in scene:
            object_selection_plan = scene["object_selection_plan"]
            if self.reuse_selection:
                selected_objects = scene["selected_objects"]
            else:
                for room_type in rooms_types:
                    floor_objects, _, wall_objects, _ = self.get_objects_by_room(
                        object_selection_plan[room_type],
                        scene,
                        room2size[room_type],
                        room2floor_capacity[room_type],
                        room2wall_capacity[room_type],
                        room2vertices[room_type],
                    )
                    selected_objects[room_type]["floor"] = floor_objects
                    selected_objects[room_type]["wall"] = wall_objects
        else:
            object_selection_plan = {room["roomType"]: [] for room in scene["rooms"]}
            packed_args = [
                (
                    room_type,
                    scene,
                    additional_requirements,
                    room2size,
                    room2floor_capacity,
                    room2wall_capacity,
                    room2vertices,
                )
                for room_type in rooms_types
            ]

            if self.multiprocessing and len(packed_args) > 1:
                pool = multiprocessing.Pool(processes=4)
                results = pool.map(self.plan_room, packed_args)
                pool.close()
                pool.join()
            else:
                results = [self.plan_room(args) for args in packed_args]

            for room_type, result in results:
                selected_objects[room_type]["floor"] = result["floor"]
                selected_objects[room_type]["wall"] = result["wall"]
                object_selection_plan[room_type] = result["plan"]

        print(
            f"\n{Fore.GREEN}AI: Here is the object selection plan:\n{object_selection_plan}{Fore.RESET}"
        )
        return object_selection_plan, selected_objects

    def plan_room(self, args):
        (
            room_type,
            scene,
            additional_requirements,
            room2size,
            room2floor_capacity,
            room2wall_capacity,
            room2vertices,
        ) = args
        print(f"\n{Fore.GREEN}AI: Selecting objects for {room_type}...{Fore.RESET}\n")

        result = {}
        room_size_str = (
            f"{int(room2size[room_type][0])*100}cm in length,"
            f" {int(room2size[room_type][1])*100}cm in width,"
            f" {int(room2size[room_type][2])*100}cm in height"
        )

        prompt_1 = (
            self.object_selection_template_1.replace("INPUT", scene["query"])
            .replace("ROOM_TYPE", room_type)
            .replace("ROOM_SIZE", room_size_str)
            .replace("REQUIREMENTS", additional_requirements)
        )

        output_1 = self.llm(prompt_1).lower()
        plan_1 = self.extract_json(output_1)

        if plan_1 is None:
            print(f"Error while extracting the JSON for {room_type}.")
            return result

        (
            floor_objects,
            floor_capacity,
            wall_objects,
            wall_capacity,
        ) = self.get_objects_by_room(
            plan_1,
            scene,
            room2size[room_type],
            room2floor_capacity[room_type],
            room2wall_capacity[room_type],
            room2vertices[room_type],
        )

        required_floor_capacity_percentage = 0.8
        if floor_capacity[1] / floor_capacity[0] >= required_floor_capacity_percentage:
            result["floor"] = floor_objects
            result["wall"] = wall_objects
            result["plan"] = plan_1
        else:
            print(
                f"{Fore.RED}AI: The used floor capacity of {room_type} is {floor_capacity[1]:.2g}m^2,"
                f" which is less than {100*required_floor_capacity_percentage:.0f}% of the total floor capacity"
                f" {floor_capacity[0]:.2g}m^2."
                f"{Fore.RESET}"
            )
            prompt_2 = self.object_selection_template_2.format(
                object_selection_prompt_new_1=prompt_1,
                object_selection_1=output_1,
                room=room_type,
            )
            output_2 = self.llm(prompt_2).lower()
            plan_2 = self.extract_json(output_2)

            if plan_2 is None:
                print(
                    f"{Fore.RED}AI: Replanning failed, will use original plan.{Fore.RESET}"
                )
                plan_2 = plan_1

            new_plan = copy.deepcopy(plan_1)
            for object in plan_2:
                new_plan[object] = plan_2[object]

            floor_objects, _, wall_objects, _ = self.get_objects_by_room(
                new_plan,
                scene,
                room2size[room_type],
                room2floor_capacity[room_type],
                room2wall_capacity[room_type],
                room2vertices[room_type],
            )

            result["floor"] = floor_objects
            result["wall"] = wall_objects
            result["plan"] = new_plan

        return room_type, result

    def extract_json(self, input_string: str) -> Optional[ObjectPlanDict]:
        # Using regex to identify the JSON structure in the string
        json_match = re.search(r"{.*}", input_string, re.DOTALL)
        if json_match:
            extracted_json = json_match.group(0)

            # Convert the extracted JSON string into a Python dictionary
            json_dict = None
            try:
                json_dict = json.loads(extracted_json)
            except:
                try:
                    json_dict = ast.literal_eval(extracted_json)
                except:
                    pass

            if json_dict is None:
                print(
                    f"{Fore.RED}[ERROR] while parsing the JSON for:\n{input_string}{Fore.RESET}",
                    flush=True,
                )
                return None

            json_dict = object_plan_from_dict(json_dict)

            return json_dict

        else:
            print(f"No valid JSON found in:\n{input_string}", flush=True)
            return None

    def get_objects_by_room(
        self,
        parsed_plan: ObjectPlanDict,
        scene,
        room_size,
        floor_capacity,
        wall_capacity,
        vertices,
    ):
        # get the floor and wall objects
        floor_object_list: List[FloorOrWallObjectDict] = []
        wall_object_list: List[FloorOrWallObjectDict] = []
        for object_name, object_info in parsed_plan.items():
            object_info["object_name"] = object_name
            if object_info["location"] == "floor":
                floor_object_list.append(object_info)
            else:
                wall_object_list.append(object_info)

        floor_objects, floor_capacity = self.get_floor_objects(
            floor_object_list, floor_capacity, room_size, vertices, scene
        )
        wall_objects, wall_capacity = self.get_wall_objects(
            wall_object_list, wall_capacity, room_size, vertices, scene
        )

        return floor_objects, floor_capacity, wall_objects, wall_capacity

    def get_room_size(self, room, wall_height):
        floor_polygon = room["floorPolygon"]
        x_values = [point["x"] for point in floor_polygon]
        z_values = [point["z"] for point in floor_polygon]
        x_dim = max(x_values) - min(x_values)
        z_dim = max(z_values) - min(z_values)

        if x_dim > z_dim:
            return (x_dim, wall_height, z_dim)
        else:
            return (z_dim, wall_height, x_dim)

    def get_room_area(self, room):
        room_vertices = room["vertices"]
        room_polygon = Polygon(room_vertices)
        return room_polygon.area

    def get_room_perimeter(self, room):
        room_vertices = room["vertices"]
        room_polygon = Polygon(room_vertices)
        return room_polygon.length

    def get_floor_objects(
        self,
        floor_object_list: List[FloorOrWallObjectDict],
        floor_capacity,
        room_size,
        room_vertices,
        scene,
    ):
        selected_floor_objects_all = []
        for floor_object in sorted(
            floor_object_list, key=lambda fo: -1 * fo["importance"]
        ):
            object_type = floor_object["object_name"]
            object_description = floor_object["description"]
            object_size = floor_object["size"]
            importance = floor_object["importance"]
            quantity = min(floor_object["quantity"], 10)

            if "variance_type" not in floor_object:
                print(
                    f'[WARNING] variance_type not found in the the object:\n{floor_object}, will set this to be "same".'
                )
            variance_type = floor_object["variance_type"]

            candidates = self.object_retriever.retrieve(
                [f"a 3D model of {object_type}, {object_description}"],
                self.similarity_threshold_floor,
            )

            candidates = [
                candidate
                for candidate, annotation in zip(
                    candidates,
                    [
                        get_annotations(self.database[candidate[0]])
                        for candidate in candidates
                    ],
                )
                if annotation["onFloor"]  # only select objects on the floor
                and (
                    not annotation["onCeiling"]
                )  # only select objects not on the ceiling
                and all(  # ignore doors and windows and frames
                    k not in annotation["category"].lower()
                    for k in ["door", "window", "frame"]
                )
            ]

            # check if the object is too big
            candidates = self.check_object_size(candidates, room_size)

            # check if object can be placed on the floor
            candidates = self.check_floor_placement(
                candidates[:20], room_vertices, scene
            )

            # No candidates found
            if len(candidates) == 0:
                print(f"No candidates found for {object_type} {object_description}")
                continue

            # remove used assets
            top_one_candidate = candidates[0]
            if len(candidates) > 1:
                candidates = [
                    candidate
                    for candidate in candidates
                    if candidate[0] not in self.used_assets
                ]
            if len(candidates) == 0:
                candidates = [top_one_candidate]

            # consider object size difference
            if object_size is not None and self.consider_size:
                candidates = self.object_retriever.compute_size_difference(
                    object_size, candidates
                )

            candidates = candidates[:10]  # only select top 10 candidates

            selected_asset_ids = []

            if variance_type == "same":
                selected_candidate = self.random_select(candidates)
                selected_asset_id = selected_candidate[0]
                selected_asset_ids = [selected_asset_id] * quantity

            elif variance_type == "varied":
                for i in range(quantity):
                    selected_candidate = self.random_select(candidates)
                    selected_asset_id = selected_candidate[0]
                    selected_asset_ids.append(selected_asset_id)
                    if len(candidates) > 1:
                        candidates.remove(selected_candidate)
            else:
                raise NotImplementedError(
                    f"Variance type {variance_type} is not supported."
                )

            for i in range(quantity):
                selected_asset_id = selected_asset_ids[i]
                object_name = f"{object_type}-{i}"
                selected_floor_objects_all.append(
                    (object_name, selected_asset_id, importance)
                )

        # reselect objects if they exceed floor capacity
        selected_floor_objects_filtered = []
        for object_name, selected_asset_id, importance in selected_floor_objects_all:
            x_size, _, z_size = get_bbox_dims_vec(self.database[selected_asset_id])
            selected_asset_area = x_size * z_size
            if (
                floor_capacity[1] + selected_asset_area > floor_capacity[0]
                and len(selected_floor_objects_filtered) > 0
            ):
                print(f"{object_name} {selected_asset_id} exceeds floor capacity")
            else:
                selected_floor_objects_filtered.append(
                    (object_name, selected_asset_id, importance)
                )
                selected_floor_objects_all.remove(
                    (object_name, selected_asset_id, importance)
                )
                floor_capacity = (
                    floor_capacity[0],
                    floor_capacity[1] + selected_asset_area,
                )

        return [
            (on, aid)
            for (on, aid, _) in sorted(
                selected_floor_objects_filtered, key=lambda x: -x[-1]
            )
        ], floor_capacity

    def get_wall_objects(
        self,
        wall_object_list: List[FloorOrWallObjectDict],
        wall_capacity,
        room_size,
        room_vertices,
        scene,
    ):
        selected_wall_objects_all = []
        for wall_object in wall_object_list:
            object_type = wall_object["object_name"]
            object_description = wall_object["description"]
            object_size = wall_object["size"]
            importance = wall_object["importance"]
            quantity = min(wall_object["quantity"], 10)
            variance_type = wall_object["variance_type"]

            candidates = self.object_retriever.retrieve(
                [f"a 3D model of {object_type}, {object_description}"],
                self.similarity_threshold_wall,
            )

            # check on wall objects
            candidates = [
                candidate
                for candidate in candidates
                if get_annotations(self.database[candidate[0]])["onWall"] == True
            ]  # only select objects on the wall

            # ignore doors and windows
            candidates = [
                candidate
                for candidate in candidates
                if "door"
                not in get_annotations(self.database[candidate[0]])["category"].lower()
            ]
            candidates = [
                candidate
                for candidate in candidates
                if "window"
                not in get_annotations(self.database[candidate[0]])["category"].lower()
            ]

            # check if the object is too big
            candidates = self.check_object_size(candidates, room_size)

            # check thin objects
            candidates = self.check_thin_object(candidates)

            # check if object can be placed on the wall
            candidates = self.check_wall_placement(
                candidates[:20], room_vertices, scene
            )

            if len(candidates) == 0:
                print(f"No candidates found for {object_type} {object_description}")
                continue

            # remove used assets
            top_one_candidate = candidates[0]
            if len(candidates) > 1:
                candidates = [
                    candidate
                    for candidate in candidates
                    if candidate[0] not in self.used_assets
                ]
            if len(candidates) == 0:
                candidates = [top_one_candidate]

            # consider object size difference
            if object_size is not None and self.consider_size:
                candidates = self.object_retriever.compute_size_difference(
                    object_size, candidates
                )

            candidates = candidates[:10]  # only select top 10 candidates

            selected_asset_ids = []
            if variance_type == "same":
                selected_candidate = self.random_select(candidates)
                selected_asset_id = selected_candidate[0]
                selected_asset_ids = [selected_asset_id] * quantity

            elif variance_type == "varied":
                for i in range(quantity):
                    selected_candidate = self.random_select(candidates)
                    selected_asset_id = selected_candidate[0]
                    selected_asset_ids.append(selected_asset_id)
                    if len(candidates) > 1:
                        candidates.remove(selected_candidate)
            else:
                raise NotImplementedError(
                    f"Variance type {variance_type} is not supported."
                )

            for i in range(quantity):
                selected_asset_id = selected_asset_ids[i]
                object_name = f"{object_type}-{i}"
                selected_wall_objects_all.append(
                    (object_name, selected_asset_id, importance)
                )

        # reselect objects if they exceed wall capacity, consider the diversity of objects
        selected_wall_objects_filtered = []
        for object_name, selected_asset_id, importance in selected_wall_objects_all:
            selected_asset_capacity, _, _ = get_bbox_dims_vec(
                self.database[selected_asset_id]
            )
            if (
                wall_capacity[1] + selected_asset_capacity > wall_capacity[0]
                and len(selected_wall_objects_filtered) > 0
            ):
                print(f"{object_name} {selected_asset_id} exceeds wall capacity")
            else:
                selected_wall_objects_filtered.append(
                    (object_name, selected_asset_id, importance)
                )
                selected_wall_objects_all.remove(
                    (object_name, selected_asset_id, importance)
                )
                wall_capacity = (
                    wall_capacity[0],
                    wall_capacity[1] + selected_asset_capacity,
                )

        return [
            (on, aid)
            for (on, aid, _) in sorted(
                selected_wall_objects_filtered, key=lambda x: -x[-1]
            )
        ], wall_capacity

    def check_object_size(self, candidates, room_size):
        valid_candidates = []
        for candidate in candidates:
            dimension = get_bbox_dims(self.database[candidate[0]])
            size = [dimension["x"], dimension["y"], dimension["z"]]
            if size[2] > size[0]:
                size = [size[2], size[1], size[0]]  # make sure that x > z

            if size[0] > room_size[0] * self.object_size_tolerance:
                continue
            if size[1] > room_size[1] * self.object_size_tolerance:
                continue
            if size[2] > room_size[2] * self.object_size_tolerance:
                continue
            if size[0] * size[2] > room_size[0] * room_size[2] * 0.5:
                continue  # TODO: consider using the floor area instead of the room area

            valid_candidates.append(candidate)

        return valid_candidates

    def check_thin_object(self, candidates):
        valid_candidates = []
        for candidate in candidates:
            dimension = get_bbox_dims(self.database[candidate[0]])
            size = [dimension["x"], dimension["y"], dimension["z"]]
            if size[2] > min(size[0], size[1]) * self.thin_threshold:
                continue
            valid_candidates.append(candidate)
        return valid_candidates

    def random_select(self, candidates):
        if self.random_selection:
            selected_candidate = random.choice(candidates)
        else:
            scores = [candidate[1] for candidate in candidates]
            scores_tensor = torch.Tensor(scores)
            probas = F.softmax(
                scores_tensor, dim=0
            )  # TODO: consider using normalized scores
            selected_index = torch.multinomial(probas, 1).item()
            selected_candidate = candidates[selected_index]
        return selected_candidate

    def update_floor_capacity(self, room2floor_capacity, scene):
        for room in scene["rooms"]:
            room_vertices = room["vertices"]
            room_poly = Polygon(room_vertices)
            for door in scene["doors"]:
                for door_vertices in door["doorBoxes"]:
                    door_poly = Polygon(door_vertices)
                    door_center = door_poly.centroid
                    door_area = door_poly.area
                    if room_poly.contains(door_center):
                        room2floor_capacity[room["id"]][1] += door_area * 0.6

            if scene["open_walls"] != []:
                for open_wall_vertices in scene["open_walls"]["openWallBoxes"]:
                    open_wall_poly = Polygon(open_wall_vertices)
                    open_wall_center = open_wall_poly.centroid
                    if room_poly.contains(open_wall_center):
                        room2floor_capacity[room["id"]][1] += open_wall_poly.area * 0.6

        return room2floor_capacity

    def update_wall_capacity(self, room2wall_capacity, scene):
        for room in scene["rooms"]:
            room_vertices = room["vertices"]
            room_poly = Polygon(room_vertices)
            for window in scene["windows"]:
                for window_vertices in window["windowBoxes"]:
                    window_poly = Polygon(window_vertices)
                    window_center = window_poly.centroid
                    window_x = window_poly.bounds[2] - window_poly.bounds[0]
                    window_y = window_poly.bounds[3] - window_poly.bounds[1]
                    window_width = max(window_x, window_y)
                    if room_poly.contains(window_center):
                        room2wall_capacity[room["id"]][1] += window_width * 0.6

            if scene["open_walls"] != []:
                for open_wall_vertices in scene["open_walls"]["openWallBoxes"]:
                    open_wall_poly = Polygon(open_wall_vertices)
                    open_wall_center = open_wall_poly.centroid
                    open_wall_x = open_wall_poly.bounds[2] - open_wall_poly.bounds[0]
                    open_wall_y = open_wall_poly.bounds[3] - open_wall_poly.bounds[1]
                    open_wall_width = max(open_wall_x, open_wall_y)
                    if room_poly.contains(open_wall_center):
                        room2wall_capacity[room["id"]][1] += open_wall_width * 0.6

        return room2wall_capacity

    def check_floor_placement(self, candidates, room_vertices, scene):
        room_x = max([vertex[0] for vertex in room_vertices]) - min(
            [vertex[0] for vertex in room_vertices]
        )
        room_z = max([vertex[1] for vertex in room_vertices]) - min(
            [vertex[1] for vertex in room_vertices]
        )
        grid_size = int(max(room_x // 20, room_z // 20))

        solver = DFS_Solver_Floor(grid_size=grid_size)

        room_poly = Polygon(room_vertices)
        initial_state = self.get_initial_state_floor(
            room_vertices, scene, add_window=False
        )

        grid_points = solver.create_grids(room_poly)
        grid_points = solver.remove_points(grid_points, initial_state)

        valid_candidates = []
        for candidate in candidates:
            object_size = get_bbox_dims(self.database[candidate[0]])
            object_dim = (
                object_size["x"] * 100 + self.size_buffer,
                object_size["z"] * 100 + self.size_buffer,
            )

            solutions = solver.get_all_solutions(room_poly, grid_points, object_dim)
            solutions = solver.filter_collision(initial_state, solutions)
            solutions = solver.place_edge(room_poly, solutions, object_dim)

            if solutions != []:
                valid_candidates.append(candidate)
            else:
                print(
                    f"Floor Object {candidate[0]} (size: {object_dim}) cannot be placed in room"
                )
                continue

        return valid_candidates

    def check_wall_placement(self, candidates, room_vertices, scene):
        room_x = max([vertex[0] for vertex in room_vertices]) - min(
            [vertex[0] for vertex in room_vertices]
        )
        room_z = max([vertex[1] for vertex in room_vertices]) - min(
            [vertex[1] for vertex in room_vertices]
        )
        grid_size = int(max(room_x // 20, room_z // 20))

        solver = DFS_Solver_Wall(grid_size=grid_size)

        room_poly = Polygon(room_vertices)
        initial_state = self.get_initial_state_wall(room_vertices, scene)
        grid_points = solver.create_grids(room_poly)

        valid_candidates = []
        for candidate in candidates:
            object_size = get_bbox_dims(self.database[candidate[0]])
            object_dim = (
                object_size["x"] * 100,
                object_size["y"] * 100,
                object_size["z"] * 100,
            )

            solutions = solver.get_all_solutions(
                room_poly, grid_points, object_dim, height=0
            )
            solutions = solver.filter_collision(initial_state, solutions)

            if solutions != []:
                valid_candidates.append(candidate)
            else:
                print(
                    f"Wall Object {candidate[0]} (size: {object_dim}) cannot be placed in room"
                )
                continue

        return valid_candidates

    def get_initial_state_floor(self, room_vertices, scene, add_window=True):
        doors, windows, open_walls = (
            scene["doors"],
            scene["windows"],
            scene["open_walls"],
        )
        room_poly = Polygon(room_vertices)

        initial_state = {}
        i = 0
        for door in doors:
            door_boxes = door["doorBoxes"]
            for door_box in door_boxes:
                door_vertices = [(x * 100, z * 100) for (x, z) in door_box]
                door_poly = Polygon(door_vertices)
                door_center = door_poly.centroid
                if room_poly.contains(door_center):
                    initial_state[f"door-{i}"] = (
                        (door_center.x, door_center.y),
                        0,
                        door_vertices,
                        1,
                    )
                    i += 1

        if add_window:
            for window in windows:
                window_boxes = window["windowBoxes"]
                for window_box in window_boxes:
                    window_vertices = [(x * 100, z * 100) for (x, z) in window_box]
                    window_poly = Polygon(window_vertices)
                    window_center = window_poly.centroid
                    if room_poly.contains(window_center):
                        initial_state[f"window-{i}"] = (
                            (window_center.x, window_center.y),
                            0,
                            window_vertices,
                            1,
                        )
                        i += 1

        if open_walls != []:
            for open_wall_box in open_walls["openWallBoxes"]:
                open_wall_vertices = [(x * 100, z * 100) for (x, z) in open_wall_box]
                open_wall_poly = Polygon(open_wall_vertices)
                open_wall_center = open_wall_poly.centroid
                if room_poly.contains(open_wall_center):
                    initial_state[f"open-{i}"] = (
                        (open_wall_center.x, open_wall_center.y),
                        0,
                        open_wall_vertices,
                        1,
                    )
                    i += 1

        return initial_state

    def get_initial_state_wall(self, room_vertices, scene):
        doors, windows, open_walls = (
            scene["doors"],
            scene["windows"],
            scene["open_walls"],
        )
        room_poly = Polygon(room_vertices)
        initial_state = {}
        i = 0
        for door in doors:
            door_boxes = door["doorBoxes"]
            for door_box in door_boxes:
                door_vertices = [(x * 100, z * 100) for (x, z) in door_box]
                door_poly = Polygon(door_vertices)
                door_center = door_poly.centroid
                if room_poly.contains(door_center):
                    door_height = door["assetPosition"]["y"] * 100 * 2
                    x_min, z_min, x_max, z_max = door_poly.bounds
                    initial_state[f"door-{i}"] = (
                        (x_min, 0, z_min),
                        (x_max, door_height, z_max),
                        0,
                        door_vertices,
                        1,
                    )
                    i += 1

        for window in windows:
            window_boxes = window["windowBoxes"]
            for window_box in window_boxes:
                window_vertices = [(x * 100, z * 100) for (x, z) in window_box]
                window_poly = Polygon(window_vertices)
                window_center = window_poly.centroid
                if room_poly.contains(window_center):
                    y_min = window["holePolygon"][0]["y"] * 100
                    y_max = window["holePolygon"][1]["y"] * 100
                    x_min, z_min, x_max, z_max = window_poly.bounds
                    initial_state[f"window-{i}"] = (
                        (x_min, y_min, z_min),
                        (x_max, y_max, z_max),
                        0,
                        window_vertices,
                        1,
                    )
                    i += 1

        if len(open_walls) != 0:
            open_wall_boxes = open_walls["openWallBoxes"]
            for open_wall_box in open_wall_boxes:
                open_wall_vertices = [(x * 100, z * 100) for (x, z) in open_wall_box]
                open_wall_poly = Polygon(open_wall_vertices)
                open_wall_center = open_wall_poly.centroid
                if room_poly.contains(open_wall_center):
                    x_min, z_min, x_max, z_max = open_wall_poly.bounds
                    initial_state[f"open-{i}"] = (
                        (x_min, 0, z_min),
                        (x_max, scene["wall_height"] * 100, z_max),
                        0,
                        open_wall_vertices,
                        1,
                    )
                    i += 1

        return initial_state
