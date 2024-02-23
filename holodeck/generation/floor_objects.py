import copy
import datetime
import json
import math
import multiprocessing
import random
import re
import time

import matplotlib.pyplot as plt
import numpy as np
from langchain import PromptTemplate, OpenAI
from rtree import index
from scipy.interpolate import interp1d
from shapely.geometry import Polygon, Point, box, LineString

import holodeck.generation.prompts as prompts
from holodeck.generation.milp_utils import *
from holodeck.generation.objaverse_retriever import ObjathorRetriever
from holodeck.generation.utils import get_bbox_dims


class FloorObjectGenerator:
    def __init__(self, object_retriever: ObjathorRetriever, llm: OpenAI):
        self.json_template = {
            "assetId": None,
            "id": None,
            "kinematic": True,
            "position": {},
            "rotation": {},
            "material": None,
            "roomId": None,
        }
        self.llm = llm
        self.object_retriever = object_retriever
        self.database = object_retriever.database
        self.constraint_prompt = PromptTemplate(
            input_variables=["room_type", "room_size", "objects"],
            template=prompts.object_constraints_prompt,
        )
        self.baseline_prompt = PromptTemplate(
            input_variables=["room_type", "room_size", "objects"],
            template=prompts.floor_baseline_prompt,
        )
        self.grid_density = 20
        self.add_window = False
        self.size_buffer = 10  # add 10 cm buffer to object size

        self.constraint_type = "llm"
        self.use_milp = False
        self.multiprocessing = False

    def generate_objects(self, scene, use_constraint=True):
        rooms = scene["rooms"]
        doors = scene["doors"]
        windows = scene["windows"]
        open_walls = scene["open_walls"]
        selected_objects = scene["selected_objects"]
        results = []

        packed_args = [
            (room, doors, windows, open_walls, selected_objects, use_constraint)
            for room in rooms
        ]
        if self.multiprocessing:
            pool = multiprocessing.Pool(processes=4)
            all_placements = pool.map(self.generate_objects_per_room, packed_args)
            pool.close()
            pool.join()
        else:
            all_placements = [
                self.generate_objects_per_room(args) for args in packed_args
            ]

        for placements in all_placements:
            results += placements

        return results

    def generate_objects_per_room(self, args):
        room, doors, windows, open_walls, selected_objects, use_constraint = args

        selected_floor_objects = selected_objects[room["roomType"]]["floor"]
        object_name2id = {
            object_name: asset_id for object_name, asset_id in selected_floor_objects
        }

        room_id = room["id"]
        room_type = room["roomType"]
        room_x, room_z = self.get_room_size(room)

        room_size = f"{room_x} cm x {room_z} cm"
        grid_size = max(room_x // self.grid_density, room_z // self.grid_density)

        object_names = list(object_name2id.keys())

        if use_constraint:
            # get constraints
            constraint_prompt = self.constraint_prompt.format(
                room_type=room_type,
                room_size=room_size,
                objects=", ".join(object_names),
            )

            if self.constraint_type == "llm":
                constraint_plan = self.llm(constraint_prompt)
            elif self.constraint_type in ["middle", "edge"]:
                constraint_plan = ""
                for object_name in object_names:
                    constraint_plan += f"{object_name} | {self.constraint_type}\n"
            else:
                print("Error: constraint type not supported!")

            print(f"plan for {room_type}: {constraint_plan}")
            constraints = self.parse_constraints(constraint_plan, object_names)

            # get objects list
            object2dimension = {
                object_name: get_bbox_dims(self.database[object_id])
                for object_name, object_id in object_name2id.items()
            }

            objects_list = [
                (
                    object_name,
                    (
                        object2dimension[object_name]["x"] * 100 + self.size_buffer,
                        object2dimension[object_name]["z"] * 100 + self.size_buffer,
                    ),
                )
                for object_name in constraints
            ]

            # get initial state
            room_vertices = [(x * 100, y * 100) for (x, y) in room["vertices"]]
            room_poly = Polygon(room_vertices)
            initial_state = self.get_door_window_placements(
                doors, windows, room_vertices, open_walls, self.add_window
            )

            # solve
            solver = DFS_Solver_Floor(
                grid_size=grid_size, max_duration=30, constraint_bouns=1
            )
            solution = solver.get_solution(
                room_poly,
                objects_list,
                constraints,
                initial_state,
                use_milp=self.use_milp,
            )
            placements = self.solution2placement(solution, object_name2id, room_id)
        else:
            object_information = ""
            for object_name in object_names:
                object_id = object_name2id[object_name]
                dimension = get_bbox_dims(self.database[object_name2id[object_name]])
                size_x = int(dimension["x"] * 100)
                size_z = int(dimension["z"] * 100)
                object_information += f"{object_name}: {size_x} cm x {size_z} cm\n"

            baseline_prompt = self.baseline_prompt.format(
                room_type=room_type,
                room_size=room_size,
                objects=", ".join(object_names),
            )
            room_origin = [
                min(v[0] for v in room["vertices"]),
                min(v[1] for v in room["vertices"]),
            ]
            all_is_placed = False
            while not all_is_placed:
                completion_text = self.llm(baseline_prompt)
                try:
                    completion_text = re.findall(
                        r"```(.*?)```", completion_text, re.DOTALL
                    )[0]
                    completion_text = re.sub(
                        r"^json", "", completion_text, flags=re.MULTILINE
                    )
                    all_data = json.loads(completion_text)
                except json.JSONDecodeError:
                    continue
                print(f"completion text for {room_type}: {completion_text}")
                placements = list()
                all_is_placed = True
                for data in all_data:
                    object_name = data["object_name"]
                    try:
                        object_id = object_name2id[object_name]
                    except KeyError:
                        all_is_placed = False
                        break

                    dimension = get_bbox_dims(
                        self.database[object_name2id[object_name]]
                    )
                    placement = self.json_template.copy()
                    placement["id"] = f"{object_name} ({room_id})"
                    placement["object_name"] = object_name
                    placement["assetId"] = object_id
                    placement["roomId"] = room_id
                    placement["position"] = {
                        "x": room_origin[0] + (data["position"]["X"] / 100),
                        "y": dimension["y"] / 2,
                        "z": room_origin[1] + (data["position"]["Y"] / 100),
                    }
                    placement["rotation"] = {"x": 0, "y": data["rotation"], "z": 0}
                    placements.append(placement)
                break  # only one iteration

        return placements

    def get_door_window_placements(
        self, doors, windows, room_vertices, open_walls, add_window=True
    ):
        room_poly = Polygon(room_vertices)
        door_window_placements = {}
        i = 0
        for door in doors:
            door_boxes = door["doorBoxes"]
            for door_box in door_boxes:
                door_vertices = [(x * 100, z * 100) for (x, z) in door_box]
                door_poly = Polygon(door_vertices)
                door_center = door_poly.centroid
                if room_poly.contains(door_center):
                    door_window_placements[f"door-{i}"] = (
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
                        door_window_placements[f"window-{i}"] = (
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
                    door_window_placements[f"open-{i}"] = (
                        (open_wall_center.x, open_wall_center.y),
                        0,
                        open_wall_vertices,
                        1,
                    )
                    i += 1

        return door_window_placements

    def get_room_size(self, room):
        floor_polygon = room["floorPolygon"]
        x_values = [point["x"] for point in floor_polygon]
        z_values = [point["z"] for point in floor_polygon]
        return (
            int(max(x_values) - min(x_values)) * 100,
            int(max(z_values) - min(z_values)) * 100,
        )

    def solution2placement(self, solutions, object_name2id, room_id):
        placements = []
        for object_name, solution in solutions.items():
            if (
                "door" in object_name
                or "window" in object_name
                or "open" in object_name
            ):
                continue
            dimension = get_bbox_dims(self.database[object_name2id[object_name]])
            placement = self.json_template.copy()
            placement["assetId"] = object_name2id[object_name]
            placement["id"] = f"{object_name} ({room_id})"
            placement["position"] = {
                "x": solution[0][0] / 100,
                "y": dimension["y"] / 2,
                "z": solution[0][1] / 100,
            }
            placement["rotation"] = {"x": 0, "y": solution[1], "z": 0}
            placement["roomId"] = room_id
            placement["vertices"] = list(solution[2])
            placement["object_name"] = object_name
            placements.append(placement)
        return placements

    def parse_constraints(self, constraint_text, object_names):
        constraint_name2type = {
            "edge": "global",
            "middle": "global",
            "in front of": "relative",
            "behind": "relative",
            "left of": "relative",
            "right of": "relative",
            "side of": "relative",
            "around": "relative",
            "face to": "direction",
            "face same as": "direction",
            "aligned": "alignment",
            "center alignment": "alignment",
            "center aligned": "alignment",
            "aligned center": "alignment",
            "edge alignment": "alignment",
            "near": "distance",
            "far": "distance",
        }

        object2constraints = {}
        plans = [plan.lower() for plan in constraint_text.split("\n") if "|" in plan]

        for plan in plans:
            # remove index
            pattern = re.compile(r"^(\d+[\.\)]\s*|- )")
            plan = pattern.sub("", plan)
            if plan[-1] == ".":
                plan = plan[:-1]

            object_name = (
                plan.split("|")[0].replace("*", "").strip()
            )  # remove * in object name

            if object_name not in object_names:
                continue

            object2constraints[object_name] = []

            constraints = plan.split("|")[1:]
            for constraint in constraints:
                constraint = constraint.strip()
                constraint_name = constraint.split(",")[0].strip()

                if constraint_name == "n/a":
                    continue

                try:
                    constraint_type = constraint_name2type[constraint_name]
                except:
                    print(f"constraint type {constraint_name} not found")
                    continue

                if constraint_type == "global":
                    object2constraints[object_name].append(
                        {"type": constraint_type, "constraint": constraint_name}
                    )
                elif constraint_type in [
                    "relative",
                    "direction",
                    "alignment",
                    "distance",
                ]:
                    try:
                        target = constraint.split(",")[1].strip()
                    except:
                        print(f"wrong format of constraint: {constraint}")
                        continue

                    if target in object2constraints:
                        if constraint_name == "around":
                            object2constraints[object_name].append(
                                {
                                    "type": "distance",
                                    "constraint": "near",
                                    "target": target,
                                }
                            )
                            object2constraints[object_name].append(
                                {
                                    "type": "direction",
                                    "constraint": "face to",
                                    "target": target,
                                }
                            )
                        elif constraint_name == "in front of":
                            object2constraints[object_name].append(
                                {
                                    "type": "relative",
                                    "constraint": "in front of",
                                    "target": target,
                                }
                            )
                            object2constraints[object_name].append(
                                {
                                    "type": "alignment",
                                    "constraint": "center aligned",
                                    "target": target,
                                }
                            )
                        else:
                            object2constraints[object_name].append(
                                {
                                    "type": constraint_type,
                                    "constraint": constraint_name,
                                    "target": target,
                                }
                            )
                    else:
                        print(
                            f"target object {target} not found in the existing constraint plan"
                        )
                        continue
                else:
                    print(f"constraint type {constraint_type} not found")
                    continue

        # clean the constraints
        object2constraints_cleaned = {}
        for object_name, constraints in object2constraints.items():
            constraints_cleaned = []
            constraint_types = []
            for constraint in constraints:
                if constraint["type"] not in constraint_types:
                    constraint_types.append(constraint["type"])
                    constraints_cleaned.append(constraint)
            object2constraints_cleaned[object_name] = constraints_cleaned

        return object2constraints

    def order_objects_by_size(self, selected_floor_objects):
        ordered_floor_objects = []
        for object_name, asset_id in selected_floor_objects:
            dimensions = get_bbox_dims(self.database[asset_id])
            size = dimensions["x"] * dimensions["z"]
            ordered_floor_objects.append([object_name, asset_id, size])
        ordered_floor_objects.sort(key=lambda x: x[2], reverse=True)
        ordered_floor_objects_no_size = [
            [object_name, asset_id]
            for object_name, asset_id, size in ordered_floor_objects
        ]
        return ordered_floor_objects_no_size


class SolutionFound(Exception):
    def __init__(self, solution):
        self.solution = solution


class DFS_Solver_Floor:
    def __init__(self, grid_size, random_seed=0, max_duration=5, constraint_bouns=0.2):
        self.grid_size = grid_size
        self.random_seed = random_seed
        self.max_duration = max_duration  # maximum allowed time in seconds
        self.constraint_bouns = constraint_bouns
        self.start_time = None
        self.solutions = []
        self.vistualize = False

        # Define the functions in a dictionary to avoid if-else conditions
        self.func_dict = {
            "global": {"edge": self.place_edge},
            "relative": self.place_relative,
            "direction": self.place_face,
            "alignment": self.place_alignment_center,
            "distance": self.place_distance,
        }

        self.constraint_type2weight = {
            "global": 1.0,
            "relative": 0.5,
            "direction": 0.5,
            "alignment": 0.5,
            "distance": 1.8,
        }

        self.edge_bouns = 0.0  # worth more than one constraint

    def get_solution(
        self, bounds, objects_list, constraints, initial_state, use_milp=False
    ):
        self.start_time = time.time()
        if use_milp:
            # iterate through the constraints list
            # for each constraint type "distance", add the same constraint to the target object
            new_constraints = constraints.copy()
            for object_name, object_constraints in constraints.items():
                for constraint in object_constraints:
                    if constraint["type"] == "distance":
                        target_object_name = constraint["target"]
                        if target_object_name in constraints.keys():
                            # if there is already a distance constraint of target object_name, continue
                            if any(
                                constraint["type"] == "distance"
                                and constraint["target"] == object_name
                                for constraint in constraints[target_object_name]
                            ):
                                continue
                            new_constraint = constraint.copy()
                            new_constraint["target"] = object_name
                            new_constraints[target_object_name].append(new_constraint)
            # iterate through the constraints list
            # for each constraint type "left of" or "right of", add the same constraint to the target object
            # for object_name, object_constraints in constraints.items():
            #    for constraint in object_constraints: if constraint["type"] == "relative":
            #        if constraint["constraint"] == "left of":
            constraints = new_constraints

            try:
                self.milp_dfs(bounds, objects_list, constraints, initial_state, 10)
            except SolutionFound as e:
                print(f"Time taken: {time.time() - self.start_time}")

        else:
            grid_points = self.create_grids(bounds)
            grid_points = self.remove_points(grid_points, initial_state)
            try:
                self.dfs(
                    bounds, objects_list, constraints, grid_points, initial_state, 30
                )
            except SolutionFound as e:
                print(f"Time taken: {time.time() - self.start_time}")

        print(f"Number of solutions found: {len(self.solutions)}")
        max_solution = self.get_max_solution(self.solutions)

        if not use_milp and self.vistualize:
            self.visualize_grid(bounds, grid_points, max_solution)

        return max_solution

    def get_max_solution(self, solutions):
        path_weights = []
        for solution in solutions:
            path_weights.append(sum([obj[-1] for obj in solution.values()]))
        max_index = np.argmax(path_weights)
        return solutions[max_index]

    def dfs(
        self,
        room_poly,
        objects_list,
        constraints,
        grid_points,
        placed_objects,
        branch_factor,
    ):
        if len(objects_list) == 0:
            self.solutions.append(placed_objects)
            return placed_objects

        if time.time() - self.start_time > self.max_duration:
            print(f"Time limit reached.")
            raise SolutionFound(self.solutions)

        object_name, object_dim = objects_list[0]
        placements = self.get_possible_placements(
            room_poly, object_dim, constraints[object_name], grid_points, placed_objects
        )

        if len(placements) == 0 and len(placed_objects) != 0:
            self.solutions.append(placed_objects)

        paths = []
        if branch_factor > 1:
            random.shuffle(placements)  # shuffle the placements of the first object

        for placement in placements[:branch_factor]:
            placed_objects_updated = copy.deepcopy(placed_objects)
            placed_objects_updated[object_name] = placement
            grid_points_updated = self.remove_points(
                grid_points, placed_objects_updated
            )

            sub_paths = self.dfs(
                room_poly,
                objects_list[1:],
                constraints,
                grid_points_updated,
                placed_objects_updated,
                1,
            )
            paths.extend(sub_paths)

        return paths

    def get_possible_placements(
        self, room_poly, object_dim, constraints, grid_points, placed_objects
    ):
        solutions = self.filter_collision(
            placed_objects, self.get_all_solutions(room_poly, grid_points, object_dim)
        )
        solutions = self.filter_facing_wall(room_poly, solutions, object_dim)
        edge_solutions = self.place_edge(
            room_poly, copy.deepcopy(solutions), object_dim
        )

        if len(edge_solutions) == 0:
            return edge_solutions

        global_constraint = next(
            (
                constraint
                for constraint in constraints
                if constraint["type"] == "global"
            ),
            None,
        )

        if global_constraint is None:
            global_constraint = {"type": "global", "constraint": "edge"}

        if global_constraint["constraint"] == "edge":
            candidate_solutions = copy.deepcopy(
                edge_solutions
            )  # edge is hard constraint
        else:
            if len(constraints) > 1:
                candidate_solutions = (
                    solutions + edge_solutions
                )  # edge is soft constraint
            else:
                candidate_solutions = copy.deepcopy(solutions)  # the first object

        candidate_solutions = self.filter_collision(
            placed_objects, candidate_solutions
        )  # filter again after global constraint

        if candidate_solutions == []:
            return candidate_solutions
        random.shuffle(candidate_solutions)
        placement2score = {
            tuple(solution[:3]): solution[-1] for solution in candidate_solutions
        }

        # add a bias to edge solutions
        for solution in candidate_solutions:
            if solution in edge_solutions and len(constraints) >= 1:
                placement2score[tuple(solution[:3])] += self.edge_bouns

        for constraint in constraints:
            if "target" not in constraint:
                continue

            func = self.func_dict.get(constraint["type"])
            valid_solutions = func(
                constraint["constraint"],
                placed_objects[constraint["target"]],
                candidate_solutions,
            )

            weight = self.constraint_type2weight[constraint["type"]]
            if constraint["type"] == "distance":
                for solution in valid_solutions:
                    bouns = solution[-1]
                    placement2score[tuple(solution[:3])] += bouns * weight
            else:
                for solution in valid_solutions:
                    placement2score[tuple(solution[:3])] += (
                        self.constraint_bouns * weight
                    )

        # normalize the scores
        for placement in placement2score:
            placement2score[placement] /= max(len(constraints), 1)

        sorted_placements = sorted(
            placement2score, key=placement2score.get, reverse=True
        )
        sorted_solutions = [
            list(placement) + [placement2score[placement]]
            for placement in sorted_placements
        ]

        return sorted_solutions

    def create_grids(self, room_poly):
        # get the min and max bounds of the room
        min_x, min_z, max_x, max_z = room_poly.bounds

        # create grid points
        grid_points = []
        for x in range(int(min_x), int(max_x), self.grid_size):
            for y in range(int(min_z), int(max_z), self.grid_size):
                point = Point(x, y)
                if room_poly.contains(point):
                    grid_points.append((x, y))

        return grid_points

    def remove_points(self, grid_points, objects_dict):
        # Create an r-tree index
        idx = index.Index()

        # Populate the index with bounding boxes of the objects
        for i, (_, _, obj, _) in enumerate(objects_dict.values()):
            idx.insert(i, Polygon(obj).bounds)

        # Create Shapely Polygon objects only once
        polygons = [Polygon(obj) for _, _, obj, _ in objects_dict.values()]

        valid_points = []

        for point in grid_points:
            p = Point(point)
            # Get a list of potential candidates
            candidates = [polygons[i] for i in idx.intersection(p.bounds)]
            # Check if point is in any of the candidate polygons
            if not any(candidate.contains(p) for candidate in candidates):
                valid_points.append(point)

        return valid_points

    def get_all_solutions(self, room_poly, grid_points, object_dim):
        obj_length, obj_width = object_dim
        obj_half_length, obj_half_width = obj_length / 2, obj_width / 2

        rotation_adjustments = {
            0: ((-obj_half_length, -obj_half_width), (obj_half_length, obj_half_width)),
            90: (
                (-obj_half_width, -obj_half_length),
                (obj_half_width, obj_half_length),
            ),
            180: (
                (-obj_half_length, obj_half_width),
                (obj_half_length, -obj_half_width),
            ),
            270: (
                (obj_half_width, -obj_half_length),
                (-obj_half_width, obj_half_length),
            ),
        }

        solutions = []
        for rotation in [0, 90, 180, 270]:
            for point in grid_points:
                center_x, center_y = point
                lower_left_adjustment, upper_right_adjustment = rotation_adjustments[
                    rotation
                ]
                lower_left = (
                    center_x + lower_left_adjustment[0],
                    center_y + lower_left_adjustment[1],
                )
                upper_right = (
                    center_x + upper_right_adjustment[0],
                    center_y + upper_right_adjustment[1],
                )
                obj_box = box(*lower_left, *upper_right)

                if room_poly.contains(obj_box):
                    solutions.append(
                        [point, rotation, tuple(obj_box.exterior.coords[:]), 1]
                    )

        return solutions

    def filter_collision(self, objects_dict, solutions):
        valid_solutions = []
        object_polygons = [
            Polygon(obj_coords) for _, _, obj_coords, _ in list(objects_dict.values())
        ]
        for solution in solutions:
            sol_obj_coords = solution[2]
            sol_obj = Polygon(sol_obj_coords)
            if not any(sol_obj.intersects(obj) for obj in object_polygons):
                valid_solutions.append(solution)
        return valid_solutions

    def filter_facing_wall(self, room_poly, solutions, obj_dim):
        valid_solutions = []
        obj_width = obj_dim[1]
        obj_half_width = obj_width / 2

        front_center_adjustments = {
            0: (0, obj_half_width),
            90: (obj_half_width, 0),
            180: (0, -obj_half_width),
            270: (-obj_half_width, 0),
        }

        valid_solutions = []
        for solution in solutions:
            center_x, center_y = solution[0]
            rotation = solution[1]

            front_center_adjustment = front_center_adjustments[rotation]
            front_center_x, front_center_y = (
                center_x + front_center_adjustment[0],
                center_y + front_center_adjustment[1],
            )

            front_center_distance = room_poly.boundary.distance(
                Point(front_center_x, front_center_y)
            )

            if front_center_distance >= 30:  # TODO: make this a parameter
                valid_solutions.append(solution)

        return valid_solutions

    def place_edge(self, room_poly, solutions, obj_dim):
        valid_solutions = []
        obj_width = obj_dim[1]
        obj_half_width = obj_width / 2

        back_center_adjustments = {
            0: (0, -obj_half_width),
            90: (-obj_half_width, 0),
            180: (0, obj_half_width),
            270: (obj_half_width, 0),
        }

        for solution in solutions:
            center_x, center_y = solution[0]
            rotation = solution[1]

            back_center_adjustment = back_center_adjustments[rotation]
            back_center_x, back_center_y = (
                center_x + back_center_adjustment[0],
                center_y + back_center_adjustment[1],
            )

            back_center_distance = room_poly.boundary.distance(
                Point(back_center_x, back_center_y)
            )
            center_distance = room_poly.boundary.distance(Point(center_x, center_y))

            if (
                back_center_distance <= self.grid_size
                and back_center_distance < center_distance
            ):
                solution[-1] += self.constraint_bouns
                # valid_solutions.append(solution) # those are still valid solutions, but we need to move the object to the edge

                # move the object to the edge
                center2back_vector = np.array(
                    [back_center_x - center_x, back_center_y - center_y]
                )
                center2back_vector /= np.linalg.norm(center2back_vector)
                offset = center2back_vector * (
                    back_center_distance + 4.5
                )  # add a small distance to avoid the object cross the wall
                solution[0] = (center_x + offset[0], center_y + offset[1])
                solution[2] = (
                    (solution[2][0][0] + offset[0], solution[2][0][1] + offset[1]),
                    (solution[2][1][0] + offset[0], solution[2][1][1] + offset[1]),
                    (solution[2][2][0] + offset[0], solution[2][2][1] + offset[1]),
                    (solution[2][3][0] + offset[0], solution[2][3][1] + offset[1]),
                )
                valid_solutions.append(solution)

        return valid_solutions

    def place_corner(self, room_poly, solutions, obj_dim):
        obj_length, obj_width = obj_dim
        obj_half_length, _ = obj_length / 2, obj_width / 2

        rotation_center_adjustments = {
            0: ((-obj_half_length, 0), (obj_half_length, 0)),
            90: ((0, obj_half_length), (0, -obj_half_length)),
            180: ((obj_half_length, 0), (-obj_half_length, 0)),
            270: ((0, -obj_half_length), (0, obj_half_length)),
        }

        edge_solutions = self.place_edge(room_poly, solutions, obj_dim)

        valid_solutions = []

        for solution in edge_solutions:
            (center_x, center_y), rotation = solution[:2]
            (dx_left, dy_left), (dx_right, dy_right) = rotation_center_adjustments[
                rotation
            ]

            left_center_x, left_center_y = center_x + dx_left, center_y + dy_left
            right_center_x, right_center_y = center_x + dx_right, center_y + dy_right

            left_center_distance = room_poly.boundary.distance(
                Point(left_center_x, left_center_y)
            )
            right_center_distance = room_poly.boundary.distance(
                Point(right_center_x, right_center_y)
            )

            if min(left_center_distance, right_center_distance) < self.grid_size:
                solution[-1] += self.constraint_bouns
                valid_solutions.append(solution)

        return valid_solutions

    def place_relative(self, place_type, target_object, solutions):
        valid_solutions = []
        _, target_rotation, target_coords, _ = target_object
        target_polygon = Polygon(target_coords)

        min_x, min_y, max_x, max_y = target_polygon.bounds
        mean_x = (min_x + max_x) / 2
        mean_y = (min_y + max_y) / 2

        comparison_dict = {
            "left of": {
                0: lambda sol_center: sol_center[0] < min_x
                and min_y <= sol_center[1] <= max_y,
                90: lambda sol_center: sol_center[1] > max_y
                and min_x <= sol_center[0] <= max_x,
                180: lambda sol_center: sol_center[0] > max_x
                and min_y <= sol_center[1] <= max_y,
                270: lambda sol_center: sol_center[1] < min_y
                and min_x <= sol_center[0] <= max_x,
            },
            "right of": {
                0: lambda sol_center: sol_center[0] > max_x
                and min_y <= sol_center[1] <= max_y,
                90: lambda sol_center: sol_center[1] < min_y
                and min_x <= sol_center[0] <= max_x,
                180: lambda sol_center: sol_center[0] < min_x
                and min_y <= sol_center[1] <= max_y,
                270: lambda sol_center: sol_center[1] > max_y
                and min_x <= sol_center[0] <= max_x,
            },
            "in front of": {
                0: lambda sol_center: sol_center[1] > max_y
                and mean_x - self.grid_size
                < sol_center[0]
                < mean_x + self.grid_size,  # in front of and centered
                90: lambda sol_center: sol_center[0] > max_x
                and mean_y - self.grid_size < sol_center[1] < mean_y + self.grid_size,
                180: lambda sol_center: sol_center[1] < min_y
                and mean_x - self.grid_size < sol_center[0] < mean_x + self.grid_size,
                270: lambda sol_center: sol_center[0] < min_x
                and mean_y - self.grid_size < sol_center[1] < mean_y + self.grid_size,
            },
            "behind": {
                0: lambda sol_center: sol_center[1] < min_y
                and min_x <= sol_center[0] <= max_x,
                90: lambda sol_center: sol_center[0] < min_x
                and min_y <= sol_center[1] <= max_y,
                180: lambda sol_center: sol_center[1] > max_y
                and min_x <= sol_center[0] <= max_x,
                270: lambda sol_center: sol_center[0] > max_x
                and min_y <= sol_center[1] <= max_y,
            },
            "side of": {
                0: lambda sol_center: min_y <= sol_center[1] <= max_y,
                90: lambda sol_center: min_x <= sol_center[0] <= max_x,
                180: lambda sol_center: min_y <= sol_center[1] <= max_y,
                270: lambda sol_center: min_x <= sol_center[0] <= max_x,
            },
        }

        compare_func = comparison_dict.get(place_type).get(target_rotation)

        for solution in solutions:
            sol_center = solution[0]

            if compare_func(sol_center):
                solution[-1] += self.constraint_bouns
                valid_solutions.append(solution)

        return valid_solutions

    def place_distance(self, distance_type, target_object, solutions):
        target_coords = target_object[2]
        target_poly = Polygon(target_coords)
        distances = []
        valid_solutions = []
        for solution in solutions:
            sol_coords = solution[2]
            sol_poly = Polygon(sol_coords)
            distance = target_poly.distance(sol_poly)
            distances.append(distance)

            solution[-1] = distance
            valid_solutions.append(solution)

        min_distance = min(distances)
        max_distance = max(distances)

        if distance_type == "near":
            if min_distance < 80:
                points = [(min_distance, 1), (80, 0), (max_distance, 0)]
            else:
                points = [(min_distance, 0), (max_distance, 0)]

        elif distance_type == "far":
            points = [(min_distance, 0), (max_distance, 1)]

        x = [point[0] for point in points]
        y = [point[1] for point in points]

        f = interp1d(x, y, kind="linear", fill_value="extrapolate")

        for solution in valid_solutions:
            distance = solution[-1]
            solution[-1] = float(f(distance))

        return valid_solutions

    def place_face(self, face_type, target_object, solutions):
        if face_type == "face to":
            return self.place_face_to(target_object, solutions)

        elif face_type == "face same as":
            return self.place_face_same(target_object, solutions)

        elif face_type == "face opposite to":
            return self.place_face_opposite(target_object, solutions)

    def place_face_to(self, target_object, solutions):
        # Define unit vectors for each rotation
        unit_vectors = {
            0: np.array([0.0, 1.0]),  # Facing up
            90: np.array([1.0, 0.0]),  # Facing right
            180: np.array([0.0, -1.0]),  # Facing down
            270: np.array([-1.0, 0.0]),  # Facing left
        }

        target_coords = target_object[2]
        target_poly = Polygon(target_coords)

        valid_solutions = []

        for solution in solutions:
            sol_center = solution[0]
            sol_rotation = solution[1]

            # Define an arbitrarily large point in the direction of the solution's rotation
            far_point = sol_center + 1e6 * unit_vectors[sol_rotation]

            # Create a half-line from the solution's center to the far point
            half_line = LineString([sol_center, far_point])

            # Check if the half-line intersects with the target polygon
            if half_line.intersects(target_poly):
                solution[-1] += self.constraint_bouns
                valid_solutions.append(solution)

        return valid_solutions

    def place_face_same(self, target_object, solutions):
        target_rotation = target_object[1]
        valid_solutions = []

        for solution in solutions:
            sol_rotation = solution[1]
            if sol_rotation == target_rotation:
                solution[-1] += self.constraint_bouns
                valid_solutions.append(solution)

        return valid_solutions

    def place_face_opposite(self, target_object, solutions):
        target_rotation = (target_object[1] + 180) % 360
        valid_solutions = []

        for solution in solutions:
            sol_rotation = solution[1]
            if sol_rotation == target_rotation:
                solution[-1] += self.constraint_bouns
                valid_solutions.append(solution)

        return valid_solutions

    def place_alignment_center(self, alignment_type, target_object, solutions):
        target_center = target_object[0]
        valid_solutions = []
        eps = 5
        for solution in solutions:
            sol_center = solution[0]
            if (
                abs(sol_center[0] - target_center[0]) < eps
                or abs(sol_center[1] - target_center[1]) < eps
            ):
                solution[-1] += self.constraint_bouns
                valid_solutions.append(solution)
        return valid_solutions

    def visualize_grid(self, room_poly, grid_points, solutions):
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 22

        # create a new figure
        fig, ax = plt.subplots()

        # draw the room
        x, y = room_poly.exterior.xy
        ax.plot(x, y, "-", label="Room", color="black", linewidth=2)

        # draw the grid points
        grid_x = [point[0] for point in grid_points]
        grid_y = [point[1] for point in grid_points]
        ax.plot(grid_x, grid_y, "o", markersize=2, color="grey")

        # draw the solutions
        for object_name, solution in solutions.items():
            center, rotation, box_coords = solution[:3]
            center_x, center_y = center

            # create a polygon for the solution
            obj_poly = Polygon(box_coords)
            x, y = obj_poly.exterior.xy
            ax.plot(x, y, "-", linewidth=2, color="black")

            # ax.text(center_x, center_y, object_name, fontsize=18, ha='center')

            # set arrow direction based on rotation
            if rotation == 0:
                ax.arrow(center_x, center_y, 0, 25, head_width=10, fc="black")
            elif rotation == 90:
                ax.arrow(center_x, center_y, 25, 0, head_width=10, fc="black")
            elif rotation == 180:
                ax.arrow(center_x, center_y, 0, -25, head_width=10, fc="black")
            elif rotation == 270:
                ax.arrow(center_x, center_y, -25, 0, head_width=10, fc="black")
        # axis off
        ax.axis("off")
        ax.set_aspect("equal", "box")  # to keep the ratios equal along x and y axis
        create_time = (
            str(datetime.datetime.now())
            .replace(" ", "-")
            .replace(":", "-")
            .replace(".", "-")
        )
        plt.savefig(f"{create_time}.pdf", bbox_inches="tight", dpi=300)
        plt.show()

    def milp_dfs(
        self, room_poly, all_objects_list, constraints, placed_objects, branch_factor=1
    ):
        if len(all_objects_list) == 0:
            self.solutions.append(placed_objects)
            return placed_objects

        if time.time() - self.start_time > self.max_duration:
            print(f"Time limit reached.")
            raise SolutionFound(self.solutions)

        def milp_solve(soft_constraints_list, hard_constraints_list, verbose=False):
            problem = cp.Problem(
                cp.Maximize(sum(soft_constraints_list)), hard_constraints_list
            )
            if verbose:
                print("solving milp using GUROBI ...")
            problem.solve(solver=cp.GUROBI, reoptimize=True, verbose=False)
            return problem.value

        def parse_object_properties(object_properties):
            x, y = object_properties[0]
            rotation = int(object_properties[1] or 0)
            # set rotation to the closest 90 degree
            rotation = int(round(rotation / 90) * 90)
            assert rotation in [0, 90, 180, 270]
            object_bbox = object_properties[2]
            min_x = min([point[0] for point in object_bbox])
            max_x = max([point[0] for point in object_bbox])
            min_y = min([point[1] for point in object_bbox])
            max_y = max([point[1] for point in object_bbox])
            object_dim = (
                (max_x - min_x, max_y - min_y)
                if rotation == 0 or rotation == 180
                else (max_y - min_y, max_x - min_x)
            )
            return x, y, rotation, object_dim

        def find_object_dim(target_object_name, objects_list, placed_objects):
            target_object_dim = None
            for object_name_1, object_dim_1 in objects_list:
                if object_name_1 == target_object_name:
                    target_object_dim = object_dim_1
                    return target_object_dim

            if not None:
                for object_name_1, object_properties in placed_objects.items():
                    if object_name_1 == target_object_name:
                        x, y, rotation, target_object_dim = parse_object_properties(
                            object_properties
                        )
                        return target_object_dim
            return None

        found_a_solution = False
        # randomly select a set of objects from all_objects_list
        # start with the largest object + more objects --> gradually reduce the number of objects
        for branch_idx in range(branch_factor):
            # sample a set of objects from a list that contains the first object

            k = random.randint(0, min(5, len(all_objects_list) - 1))
            objects_list = [all_objects_list[0]] + random.sample(
                all_objects_list[1:], k
            )

            hard_constraints_list = []
            soft_constraints_list = [0]

            # formulate the milp problem
            # object_name, object_dim = objects_list[0]
            # x, y, rotate_180, rotate_90
            variables_dict = {
                object[0]: [
                    cp.Variable(),
                    cp.Variable(),
                    cp.Variable(boolean=True),
                    cp.Variable(boolean=True),
                ]
                for object in objects_list
            }
            # add placed objects into variables dict even though they are not variables
            for object, object_properties in placed_objects.items():
                x, y = object_properties[0]
                rotation = int(object_properties[1])
                variables_dict[object] = [
                    x,
                    y,
                    rotation == 180,
                    rotation == 90 or rotation == 270,
                ]

            # Initialize a list of variables, each variable represents the coordinate for each object
            room_min_x, room_min_y, room_max_x, room_max_y = room_poly.bounds
            # Add boundary constraints to all objects
            for object_name, object_dim in objects_list:
                hard_constraints_list.extend(
                    create_boundary_constraints(
                        variables_dict[object_name],
                        object_dim,
                        (room_min_x, room_min_y, room_max_x, room_max_y),
                    )
                )
            # Add pariwise collision constraints
            for object_name_1, object_dim_1 in objects_list:
                for object_name_2, object_dim_2 in objects_list:
                    if object_name_1 == object_name_2:
                        continue
                    # collision constraints should be hard constraints
                    hard_constraints_list.extend(
                        create_nooverlap_constraints(
                            variables_dict[object_name_1],
                            variables_dict[object_name_2],
                            object_dim_1,
                            object_dim_2,
                        )
                    )

            # Add pariwise collision constraints with placed objects
            for object_name_1, object_dim_1 in objects_list:
                for object_name_2, object_properties_2 in placed_objects.items():
                    # bbox is a list of four points
                    x, y, rotation, object_dim_2 = parse_object_properties(
                        object_properties_2
                    )

                    hard_constraints_list.extend(
                        create_nooverlap_constraints(
                            variables_dict[object_name_1],
                            [x, y, rotation == 180, rotation == 90 or rotation == 270],
                            object_dim_1,
                            object_dim_2,
                        )
                    )

            # default constraints / heuristics?
            for object_name, object_dim in objects_list:
                # encourage dispersement of assets
                all_other_objects_list = [
                    x[0] for x in objects_list if x[0] != object_name
                ] + list(placed_objects.keys())
                for target_object_name in all_other_objects_list:
                    hard_constraints, soft_constraints = create_distance_constraints(
                        variables_dict[object_name],
                        variables_dict[target_object_name],
                        upper_bound=[room_max_x - room_min_x, room_max_y - room_min_y],
                        type="far",
                    )
                    assert len(soft_constraints) == 1
                    # soft_constraints[0] *= 0.001
                    hard_constraints_list.extend(hard_constraints)
                    soft_constraints_list.extend(soft_constraints)

            # use cvxpy to solve for the hard constraints
            for object_name, object_dim in objects_list:

                # by default - add soft edge constraints although this might make the solver take a longer time
                if not any(
                    constraint["type"] == "global"
                    for constraint in constraints[object_name]
                ):
                    hard_constraints, soft_constraints = create_edge_constraints(
                        variables_dict[object_name],
                        object_dim,
                        room_dim=(room_min_x, room_min_y, room_max_x, room_max_y),
                        hard=False,
                    )
                    soft_constraints[0] *= 100
                    hard_constraints_list.extend(hard_constraints)
                    soft_constraints_list.extend(soft_constraints)

                for constraint in constraints[object_name]:
                    if constraint["type"] == "global":
                        if constraint["constraint"] == "edge":  # hard constraints
                            hard_constraints, soft_constraints = (
                                create_edge_constraints(
                                    variables_dict[object_name],
                                    object_dim,
                                    room_dim=(
                                        room_min_x,
                                        room_min_y,
                                        room_max_x,
                                        room_max_y,
                                    ),
                                    hard=True,
                                )
                            )
                            hard_constraints_list.extend(hard_constraints)
                            soft_constraints_list.extend(soft_constraints)

                    if constraint["type"] == "direction":
                        assert constraint["constraint"] == "face to"
                        target_object_name = constraint["target"]
                        target_object_dim = find_object_dim(
                            target_object_name, objects_list, placed_objects
                        )
                        if target_object_dim:
                            hard_constraints_list.extend(
                                create_directional_constraints(
                                    variables_dict[object_name],
                                    variables_dict[target_object_name],
                                    object_dim,
                                    target_object_dim,
                                )
                            )

                    if constraint["type"] == "alignment":
                        assert constraint["constraint"] == "center aligned"
                        target_object_name = constraint["target"]
                        target_object_dim = find_object_dim(
                            target_object_name, objects_list, placed_objects
                        )
                        if target_object_dim:
                            hard_constraints_list.extend(
                                create_alignment_constraints(
                                    variables_dict[object_name],
                                    variables_dict[target_object_name],
                                    object_dim,
                                    target_object_dim,
                                )
                            )

                    if constraint["type"] == "distance":
                        target_object_name = constraint["target"]
                        target_object_dim = find_object_dim(
                            target_object_name, objects_list, placed_objects
                        )
                        if target_object_dim:
                            hard_constraints, soft_constraints = (
                                create_distance_constraints(
                                    variables_dict[object_name],
                                    variables_dict[target_object_name],
                                    upper_bound=[
                                        room_max_x - room_min_x,
                                        room_max_y - room_min_y,
                                    ],
                                    type=constraint["constraint"],
                                )
                            )
                            hard_constraints_list.extend(hard_constraints)
                            soft_constraints_list.extend(soft_constraints)
                            assert len(soft_constraints) == 1
                            # higher weighting
                            soft_constraints[0] *= 0.01

                    if constraint["type"] == "relative":
                        target_object_name = constraint["target"]
                        target_object_dim = find_object_dim(
                            target_object_name, objects_list, placed_objects
                        )
                        if target_object_dim:
                            hard_constraints_list.extend(
                                create_relative_constraints(
                                    variables_dict[object_name],
                                    variables_dict[target_object_name],
                                    object_dim,
                                    target_object_dim,
                                    constraint["constraint"],
                                )
                            )

            result = milp_solve(
                soft_constraints_list, hard_constraints_list, verbose=False
            )
            if result is None or math.isnan(result) or math.isinf(result):
                continue

            found_a_solution = True
            print(result, [x[0] for x in objects_list])

            # we fonud a valid solution
            # convert the placements to the same format as the dfs solver
            placed_objects_updated = copy.deepcopy(placed_objects)
            for object_name, object_dim in objects_list:
                # (x, y), rotation, bbox, score
                x = variables_dict[object_name][0].value.item()
                y = variables_dict[object_name][1].value.item()
                rotate_180 = variables_dict[object_name][2].value
                rotate_90 = variables_dict[object_name][3].value
                if not rotate_180:
                    rotate_180 = 0
                if not rotate_90:
                    rotate_90 = 0

                # bbox has taken into account of the rotation
                if rotate_90:
                    bbox = [
                        (x - object_dim[1] / 2, y - object_dim[0] / 2),
                        (x + object_dim[1] / 2, y - object_dim[0] / 2),
                        (x + object_dim[1] / 2, y + object_dim[0] / 2),
                        (x - object_dim[1] / 2, y + object_dim[0] / 2),
                    ]
                else:
                    bbox = [
                        (x - object_dim[0] / 2, y - object_dim[1] / 2),
                        (x + object_dim[0] / 2, y - object_dim[1] / 2),
                        (x + object_dim[0] / 2, y + object_dim[1] / 2),
                        (x - object_dim[0] / 2, y + object_dim[1] / 2),
                    ]

                placed_objects_updated[object_name] = [
                    (x, y),
                    rotate_180 * 180 + rotate_90 * 90,
                    bbox,
                    len(constraints[object_name]),
                ]

            # remove all elemnts in objects_list from all_objects_list
            self.milp_dfs(
                room_poly,
                [x for x in all_objects_list if x not in objects_list],
                constraints,
                placed_objects_updated,
                branch_factor=1,
            )

        if not found_a_solution and len(placed_objects) != 0:
            self.solutions.append(placed_objects)

    def test_dfs_placement(self):
        room_vertices = ((0, 0), (0, 500), (500, 500), (500, 0))
        room_poly = Polygon(room_vertices)
        grid_points = self.create_grids(room_poly)
        objects = {"door": ((50, 50), 0, ((0, 0), (100, 0), (100, 100), (0, 100)), 1)}
        grid_points = self.remove_points(grid_points, objects)
        # self.visualize_grid(room_poly, grid_points, objects)

        object_dim = (200, 100)
        solutions = self.get_all_solutions(room_poly, grid_points, object_dim)
        solutions = self.filter_collision(objects, solutions)
        solutions = self.place_edge(room_poly, solutions, object_dim)

        # for i, solution in enumerate(solutions):
        #     objects[f"sofa-{i}"] = solution
        # self.visualize_grid(room_poly, grid_points, objects)

        random.seed(0)
        objects["sofa"] = random.choice(solutions)
        # self.visualize_grid(room_poly, grid_points, objects)
        object_1_dim = (100, 50)

        solutions_1 = self.get_all_solutions(room_poly, grid_points, object_1_dim)
        solutions_1 = self.filter_collision(objects, solutions_1)

        # random.seed(42)
        # for i, solution in enumerate(random.sample(solutions_1, 25)):
        #     objects[f"coffee table-{i}"] = solution

        # objects[f"coffee table"] = [(300, 350), 0, ((350.0, 325.0), (350.0, 375.0), (250.0, 375.0), (250.0, 325.0), (350.0, 325.0)), 1.0]
        # self.visualize_grid(room_poly, grid_points, objects)

        solutions_1 = self.place_face_to(objects["sofa"], solutions_1)
        solutions_1 = self.place_relative("in front of", objects["sofa"], solutions_1)
        solutions_1 = self.place_alignment_center(
            "center alignment", objects["sofa"], solutions_1
        )
        solutions_1 = self.place_distance("near", objects["sofa"], solutions_1)
        objects[f"coffee table"] = solutions_1[-1]
        self.visualize_grid(room_poly, grid_points, objects)

    def test_milp_placement(self, simple=False, use_milp=True):
        room_vertices = ((0, 0), (0, 600), (800, 600), (800, 0))
        room_poly = Polygon(room_vertices)
        grid_points = self.create_grids(room_poly)

        if not simple:
            constraints = {
                "sofa-0": [{"type": "global", "constraint": "edge"}],
                "sofa-1": [
                    {"type": "global", "constraint": "edge"},
                    {"type": "distance", "constraint": "near", "target": "sofa-0"},
                    {
                        "type": "alignment",
                        "constraint": "center aligned",
                        "target": "sofa-0",
                    },
                ],
                "tv stand-0": [
                    {"type": "global", "constraint": "edge"},
                    {"type": "distance", "constraint": "far", "target": "sofa-1"},
                    {
                        "type": "alignment",
                        "constraint": "center aligned",
                        "target": "sofa-1",
                    },
                ],
                "coffee table-0": [
                    {"type": "global", "constraint": "middle"},
                    {"type": "distance", "constraint": "near", "target": "sofa-0"},
                    {
                        "type": "relative",
                        "constraint": "in front of",
                        "target": "sofa-0",
                    },
                    {
                        "type": "alignment",
                        "constraint": "center aligned",
                        "target": "sofa-0",
                    },
                    {
                        "type": "alignment",
                        "constraint": "center aligned",
                        "target": "sofa-0",
                    },
                    {
                        "type": "direction",
                        "constraint": "face to",
                        "target": "tv stand-0",
                    },
                ],
                "coffee table-1": [
                    {"type": "global", "constraint": "middle"},
                    {"type": "distance", "constraint": "near", "target": "sofa-1"},
                    {
                        "type": "relative",
                        "constraint": "in front of",
                        "target": "sofa-1",
                    },
                    {
                        "type": "alignment",
                        "constraint": "center aligned",
                        "target": "sofa-1",
                    },
                    {
                        "type": "alignment",
                        "constraint": "center aligned",
                        "target": "sofa-1",
                    },
                    {
                        "type": "direction",
                        "constraint": "face to",
                        "target": "tv stand-0",
                    },
                ],
                "side table-0": [
                    {"type": "global", "constraint": "edge"},
                    {"type": "distance", "constraint": "near", "target": "sofa-0"},
                    {"type": "relative", "constraint": "side of", "target": "sofa-0"},
                ],
                "side table-1": [
                    {"type": "global", "constraint": "edge"},
                    {"type": "distance", "constraint": "near", "target": "sofa-1"},
                    {"type": "relative", "constraint": "side of", "target": "sofa-1"},
                ],
                "armchair-0": [
                    {"type": "global", "constraint": "middle"},
                    {
                        "type": "distance",
                        "constraint": "near",
                        "target": "coffee table-0",
                    },
                    {
                        "type": "direction",
                        "constraint": "face to",
                        "target": "coffee table-0",
                    },
                    {
                        "type": "direction",
                        "constraint": "face to",
                        "target": "coffee table-0",
                    },
                ],
                "armchair-1": [
                    {"type": "global", "constraint": "middle"},
                    {
                        "type": "distance",
                        "constraint": "near",
                        "target": "coffee table-1",
                    },
                    {
                        "type": "direction",
                        "constraint": "face to",
                        "target": "coffee table-1",
                    },
                    {
                        "type": "direction",
                        "constraint": "face to",
                        "target": "coffee table-1",
                    },
                ],
                "bookshelf-0": [
                    {"type": "global", "constraint": "edge"},
                    {"type": "distance", "constraint": "far", "target": "tv stand-0"},
                ],
                "bookshelf-1": [
                    {"type": "global", "constraint": "edge"},
                    {"type": "distance", "constraint": "far", "target": "bookshelf-0"},
                    {
                        "type": "alignment",
                        "constraint": "center aligned",
                        "target": "bookshelf-0",
                    },
                ],
            }

            initial_state = {
                "door-0": (
                    (586.7550200520433, 550.0),
                    0,
                    [
                        (640.8300346432603, 500.0),
                        (532.6800054608262, 500.0),
                        (532.6800054608262, 600.0),
                        (640.8300346432603, 600.0),
                    ],
                    1,
                )
            }

            objects = [
                ("sofa-0", (301.6667297651499, 106.48952360032415)),
                ("sofa-1", (301.6667297651499, 106.48952360032415)),
                ("tv stand-0", (201.0964714933229, 59.39910836195032)),
                ("coffee table-0", (69.15754261308616, 126.69169450358964)),
                ("coffee table-1", (69.15754261308616, 126.69169450358964)),
                ("side table-0", (61.74632023132328, 61.74453745262855)),
                ("side table-1", (61.74632023132328, 61.74453745262855)),
                ("armchair-0", (79.0368498902692, 89.4893987892571)),
                ("armchair-1", (79.0368498902692, 89.4893987892571)),
                ("bookshelf-0", (67.94689517917222, 43.8934937031396)),
                ("bookshelf-1", (67.94689517917222, 43.8934937031396)),
            ]
            solution = self.get_solution(
                room_poly, objects, constraints, initial_state, use_milp=use_milp
            )
        else:
            constraints = {
                "dining table": [
                    {"type": "global", "constraint": "edge"},
                    {"type": "distance", "constraint": "far", "target": "door"},
                    {"type": "distance", "constraint": "near", "target": "chair"},
                ],
                "chair": [
                    {
                        "type": "relative",
                        "constraint": "side of",
                        "target": "dining table",
                    }
                ],
            }
            initial_state = {
                "door": ((50, 50), 0, ((0, 0), (100, 0), (100, 100), (0, 100)), 1)
            }
            objects = [("dining table", (100, 50)), ("chair", (50, 50))]
            solution = self.get_solution(
                room_poly, objects, constraints, initial_state, use_milp=use_milp
            )

        print("milp solution:", len(solution))
        for object_name, object_properties in solution.items():
            print(object_name, object_properties)
            # if object_properties[2] == 90 or object_properties[2] == 270:
        self.visualize_grid(room_poly, grid_points, solution)


if __name__ == "__main__":
    solver = DFS_Solver_Floor(max_duration=30, grid_size=50)
    solver.test_dfs_placement()
    solver.test_milp_placement(simple=False, use_milp=True)
