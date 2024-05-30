import copy
import multiprocessing
import random

import torch
import torch.nn.functional as F
from ai2thor.controller import Controller
from ai2thor.hooks.procedural_asset_hook import ProceduralAssetHookRunner
from langchain import OpenAI
from procthor.constants import FLOOR_Y
from procthor.utils.types import Vector3

from ai2holodeck.constants import THOR_COMMIT_ID
from ai2holodeck.generation.objaverse_retriever import ObjathorRetriever
from ai2holodeck.generation.utils import (
    get_bbox_dims,
    get_annotations,
    get_secondary_properties,
)


class SmallObjectGenerator:
    def __init__(self, object_retriever: ObjathorRetriever, llm: OpenAI):
        self.llm = llm
        self.object_retriever = object_retriever
        self.database = object_retriever.database

        # set kinematic to false for small objects
        self.json_template = {
            "assetId": None,
            "id": None,
            "kinematic": False,
            "position": {},
            "rotation": {},
            "material": None,
            "roomId": None,
        }
        self.clip_threshold = 30

        self.used_assets = []
        self.reuse_assets = True

    def generate_small_objects(self, scene, controller, receptacle_ids):
        object_selection_plan = scene["object_selection_plan"]

        receptacle2asset_id = self.get_receptacle2asset_id(scene, receptacle_ids)
        # receptacle2rotation = self.get_receptacle2rotation(scene, receptacle_ids)
        # receptacle2position = self.get_receptacle2position(scene, receptacle_ids)

        if "receptacle2small_objects" in scene and self.reuse_assets:
            receptacle2small_objects = scene["receptacle2small_objects"]
        else:
            receptacle2small_objects = self.select_small_objects(
                object_selection_plan, receptacle_ids, receptacle2asset_id
            )

        results = []
        # Place the objects
        for receptacle, small_objects in receptacle2small_objects.items():
            placements = []
            for object_name, asset_id, _ in small_objects:
                thin, rotation = self.check_thin_asset(asset_id)
                small, y_rotation = self.check_small_asset(
                    asset_id
                )  # check if the object is small and rotate around y axis randomly

                obj = self.place_object(controller, asset_id, receptacle, rotation)

                if obj != None:  # If the object is successfully placed
                    placement = self.json_template.copy()
                    placement["assetId"] = asset_id
                    placement["id"] = f"{object_name}|{receptacle}"
                    placement["position"] = obj["position"]
                    asset_height = get_bbox_dims(self.database[asset_id])["y"]

                    if obj["position"]["y"] + asset_height > scene["wall_height"]:
                        continue  # if the object is too high, skip it

                    placement["position"]["y"] = (
                        obj["position"]["y"] + (asset_height / 2) + 0.001
                    )  # add half of the height to the y position and a small offset
                    placement["rotation"] = obj["rotation"]
                    placement["roomId"] = receptacle.split("(")[1].split(")")[0]

                    # temporary solution fix position and rotation for thin objects
                    if thin:
                        placement = self.fix_placement_for_thin_assets(placement)

                    if small:
                        placement["rotation"][
                            "y"
                        ] = y_rotation  # temporary solution for random rotation around y axis for small objects
                    # else: placement["rotation"]["y"] = receptacle2rotation[receptacle]["y"]

                    if not small and not thin:
                        placement["kinematic"] = (
                            True  # set kinematic to true for non-small objects
                        )

                    if "CanBreak" in get_secondary_properties(self.database[asset_id]):
                        placement["kinematic"] = True

                    placements.append(placement)

            # TODO: check collision between small objects on the same receptacle
            valid_placements = self.check_collision(placements)
            results.extend(valid_placements)

        controller.stop()
        return results, receptacle2small_objects

    def get_receptacle2asset_id(self, scene, receptacle_ids):
        receptacle2asset_id = {}
        for object in scene["objects"]:
            receptacle2asset_id[object["id"]] = object["assetId"]
        # for receptacle_id in receptacle_ids:
        #     if receptacle_id not in receptacle2asset_id and "___" in receptacle_id:
        #         receptacle2asset_id[receptacle_id] = receptacle2asset_id[receptacle_id.split("___")[0]]
        return receptacle2asset_id

    def get_receptacle2rotation(self, scene, receptacle_ids):
        receptacle2rotation = {}
        for object in scene["objects"]:
            receptacle2rotation[object["id"]] = object["rotation"]
        # for receptacle_id in receptacle_ids:
        #     if receptacle_id not in receptacle2rotation and "___" in receptacle_id:
        #         receptacle2rotation[receptacle_id] = receptacle2rotation[receptacle_id.split("___")[0]]
        return receptacle2rotation

    def get_receptacle2position(self, scene, receptacle_ids):
        receptacle2rotation = {}
        for object in scene["objects"]:
            receptacle2rotation[object["id"]] = object["position"]
        # for receptacle_id in receptacle_ids:
        #     if receptacle_id not in receptacle2rotation and "___" in receptacle_id:
        #         receptacle2rotation[receptacle_id] = receptacle2rotation[receptacle_id.split("___")[0]]
        return receptacle2rotation

    def select_small_objects(
        self, object_selection_plan, recpetacle_ids, receptacle2asset_id
    ):
        children_plans = []
        for room_type, objects in object_selection_plan.items():
            for object_name, object_info in objects.items():
                for child in object_info["objects_on_top"]:
                    child_plan = copy.deepcopy(child)
                    child_plan["room_type"] = room_type
                    child_plan["parent"] = object_name
                    children_plans.append(child_plan)

        receptacle2small_object_plans = {}
        for receptacle_id in recpetacle_ids:
            small_object_plans = []

            for child_plan in children_plans:
                if (
                    child_plan["room_type"] in receptacle_id
                    and child_plan["parent"] in receptacle_id
                ):
                    small_object_plans.append(child_plan)

            if len(small_object_plans) > 0:
                receptacle2small_object_plans[receptacle_id] = small_object_plans

        receptacle2small_objects = {}
        packed_args = [
            (receptacle, small_objects, receptacle2asset_id)
            for receptacle, small_objects in receptacle2small_object_plans.items()
        ]
        pool = multiprocessing.Pool(processes=4)
        results = pool.map(self.select_small_objects_per_receptacle, packed_args)
        pool.close()
        pool.join()

        for result in results:
            receptacle2small_objects[result[0]] = result[1]

        return receptacle2small_objects

    def select_small_objects_per_receptacle(self, args):
        receptacle, small_objects, receptacle2asset_id = args

        results = []
        receptacle_dimensions = get_bbox_dims(
            self.database[receptacle2asset_id[receptacle]]
        )
        receptacle_size = [receptacle_dimensions["x"], receptacle_dimensions["z"]]
        receptacle_area = receptacle_size[0] * receptacle_size[1]
        capacity = 0
        num_objects = 0
        sorted(receptacle_size)
        for small_object in small_objects:
            object_name, quantity, variance_type = (
                small_object["object_name"],
                small_object["quantity"],
                small_object["variance_type"],
            )
            quantity = min(quantity, 5)  # maximum 5 objects per receptacle
            print(f"Selecting {quantity} {object_name} for {receptacle}")
            # Select the object
            candidates = self.object_retriever.retrieve(
                [f"a 3D model of {object_name}"], self.clip_threshold
            )
            candidates = [
                candidate
                for candidate in candidates
                if get_annotations(self.database[candidate[0]])["onObject"] == True
            ]  # Only select objects that can be placed on other objects

            valid_candidates = []  # Only select objects with high confidence

            for candidate in candidates:
                candidate_dimensions = get_bbox_dims(self.database[candidate[0]])
                candidate_size = [candidate_dimensions["x"], candidate_dimensions["z"]]
                sorted(candidate_size)
                if (
                    candidate_size[0] < receptacle_size[0] * 0.9
                    and candidate_size[1] < receptacle_size[1] * 0.9
                ):  # if the object is smaller than the receptacle, threshold is 90%
                    valid_candidates.append(candidate)

            if len(valid_candidates) == 0:
                print(f"No valid candidate for {object_name}.")
                continue

            # remove used assets
            top_one_candidate = valid_candidates[0]
            if len(valid_candidates) > 1:
                valid_candidates = [
                    candidate
                    for candidate in valid_candidates
                    if candidate[0] not in self.used_assets
                ]
            if len(valid_candidates) == 0:
                valid_candidates = [top_one_candidate]

            valid_candidates = valid_candidates[:5]  # only select top 5 candidates

            selected_asset_ids = []
            if variance_type == "same":
                selected_candidate = self.random_select(valid_candidates)
                selected_asset_id = selected_candidate[0]
                selected_asset_ids = [selected_asset_id] * quantity

            elif variance_type == "varied":
                for i in range(quantity):
                    selected_candidate = self.random_select(valid_candidates)
                    selected_asset_id = selected_candidate[0]
                    selected_asset_ids.append(selected_asset_id)
                    if len(valid_candidates) > 1:
                        valid_candidates.remove(selected_candidate)

            for i in range(quantity):
                small_object_dimensions = get_bbox_dims(
                    self.database[selected_asset_ids[i]]
                )
                small_object_sizes = [
                    small_object_dimensions["x"],
                    small_object_dimensions["y"],
                    small_object_dimensions["z"],
                ]
                sorted(small_object_sizes)
                # small_object_area = small_object_dimensions["x"] * small_object_dimensions["z"]
                # take the maximum 2 dimensions and multiply them
                small_object_area = small_object_sizes[1] * small_object_sizes[2] * 0.8
                capacity += small_object_area
                num_objects += 1
                if capacity > receptacle_area * 0.9 and num_objects > 1:
                    print(f"Warning: {receptacle} is overfilled.")
                    break
                if num_objects > 15:
                    print(f"Warning: {receptacle} has too many objects.")
                    break
                else:
                    results.append((f"{object_name}-{i}", selected_asset_ids[i]))

        ordered_small_objects = []
        for object_name, asset_id in results:
            dimensions = get_bbox_dims(self.database[asset_id])
            size = max(dimensions["x"], dimensions["z"])
            ordered_small_objects.append((object_name, asset_id, size))
        ordered_small_objects.sort(key=lambda x: x[2], reverse=True)

        return receptacle, ordered_small_objects

    def start_controller(self, scene, objaverse_dir):
        controller = Controller(
            commit_id=THOR_COMMIT_ID,
            agentMode="default",
            makeAgentsVisible=False,
            visibilityDistance=1.5,
            scene=scene,
            width=224,
            height=224,
            fieldOfView=40,
            action_hook_runner=ProceduralAssetHookRunner(
                asset_directory=objaverse_dir,
                asset_symlink=True,
                verbose=True,
            ),
        )
        return controller

    def place_object(self, controller, object_id, receptacle_id, rotation=[0, 0, 0]):
        generated_id = f"small|{object_id}"
        # Spawn the object
        event = controller.step(
            action="SpawnAsset",
            assetId=object_id,
            generatedId=generated_id,
            position=Vector3(x=0, y=FLOOR_Y - 20, z=0),
            rotation=Vector3(x=0, y=0, z=0),
            renderImage=False,
        )

        # Place the object in the receptacle
        # Question: Can I spawn multiple objects at once?
        event = controller.step(
            action="InitialRandomSpawn",
            randomSeed=random.randint(0, 1_000_000_000),
            objectIds=[generated_id],
            receptacleObjectIds=[receptacle_id],
            forceVisible=False,
            allowFloor=False,
            renderImage=False,
            allowMoveable=True,
            numPlacementAttempts=10,  # TODO: need to find a better way to determine the number of placement attempts
        )

        obj = next(
            obj for obj in event.metadata["objects"] if obj["objectId"] == generated_id
        )
        center_position = obj["axisAlignedBoundingBox"]["center"].copy()

        if event and center_position["y"] > FLOOR_Y:
            return obj
        else:
            controller.step(
                action="DisableObject",
                objectId=generated_id,
                renderImage=False,
            )
            return None

    def check_thin_asset(self, asset_id):
        dimensions = get_bbox_dims(self.database[asset_id])
        twod_size = (dimensions["x"] * 100, dimensions["z"] * 100)
        threshold = 5  # 3cm is the threshold for thin objects # TODO: need a better way to determine thin threshold

        rotations = [0, 0, 0]
        if twod_size[0] < threshold:
            rotations = [0, 90, 0]  # asset is thin in x direction
            return True, rotations

        elif twod_size[1] < threshold:
            rotations = [90, 0, 0]  # asset is thin in z direction
            return True, rotations

        else:
            return False, rotations

    def fix_placement_for_thin_assets(self, placement):
        asset_id = placement["assetId"]
        dimensions = get_bbox_dims(self.database[asset_id])
        threshold = 0.03  # 0.03 meter is the threshold for thin objects

        orginal_rotation = placement["rotation"]
        orginal_position = placement["position"]
        bottom_center_position = {
            "x": orginal_position["x"],
            "y": orginal_position["y"] - dimensions["y"] / 2,
            "z": orginal_position["z"],
        }

        if dimensions["x"] <= threshold:
            # asset is thin in x direction, need to rotate in z direction
            placement["rotation"] = {
                "x": orginal_rotation["x"],
                "y": orginal_rotation["y"],
                "z": orginal_rotation["z"] + 90,
            }
            placement["position"] = {
                "x": bottom_center_position["x"],
                "y": bottom_center_position["y"] + dimensions["x"] / 2,
                "z": bottom_center_position["z"],
            }

        elif dimensions["z"] <= threshold:
            # asset is thin in z direction, need to rotate in x direction
            placement["rotation"] = {
                "x": orginal_rotation["x"] + 90,
                "y": orginal_rotation["y"],
                "z": orginal_rotation["z"],
            }
            placement["position"] = {
                "x": bottom_center_position["x"],
                "y": bottom_center_position["y"] + dimensions["z"] / 2,
                "z": bottom_center_position["z"],
            }

        return placement

    def check_small_asset(self, asset_id):
        dimensions = get_bbox_dims(self.database[asset_id])
        size = (dimensions["x"] * 100, dimensions["y"] * 100, dimensions["z"] * 100)
        threshold = 25 * 25  # 25cm * 25cm is the threshold for small objects

        if (
            size[0] * size[2] <= threshold
            and size[0] <= 25
            and size[1] <= 25
            and size[2] <= 25
        ):
            return True, random.randint(0, 360)
        else:
            return False, 0

    def random_select(self, candidates):
        scores = [candidate[1] for candidate in candidates]
        scores_tensor = torch.Tensor(scores)
        probas = F.softmax(
            scores_tensor, dim=0
        )  # TODO: consider using normalized scores
        selected_index = torch.multinomial(probas, 1).item()
        selected_candidate = candidates[selected_index]
        return selected_candidate

    def check_collision(self, placements):
        static_placements = [
            placement for placement in placements if placement["kinematic"] == True
        ]

        if len(static_placements) <= 1:
            return placements
        else:
            colliding_pairs = []
            for i, placement_1 in enumerate(static_placements[:-1]):
                for placement_2 in static_placements[i + 1 :]:
                    box1 = self.get_bounding_box(placement_1)
                    box2 = self.get_bounding_box(placement_2)
                    if self.intersect_3d(box1, box2):
                        colliding_pairs.append((placement_1["id"], placement_2["id"]))
            id2assetId = {
                placement["id"]: placement["assetId"] for placement in placements
            }
            if len(colliding_pairs) != 0:
                remove_ids = []
                colliding_ids = list(
                    set(
                        [pair[0] for pair in colliding_pairs]
                        + [pair[1] for pair in colliding_pairs]
                    )
                )
                # order by size from small to large
                colliding_ids = sorted(
                    colliding_ids,
                    key=lambda x: get_bbox_dims(self.database[id2assetId[x]])["x"]
                    * get_bbox_dims(self.database[id2assetId[x]])["z"],
                )
                for object_id in colliding_ids:
                    remove_ids.append(object_id)
                    colliding_pairs = [
                        pair for pair in colliding_pairs if object_id not in pair
                    ]
                    if len(colliding_pairs) == 0:
                        break
                valid_placements = [
                    placement
                    for placement in placements
                    if placement["id"] not in remove_ids
                ]
                return valid_placements
            else:
                return placements

    def get_bounding_box(self, placement):
        asset_id = placement["assetId"]
        dimensions = get_bbox_dims(self.database[asset_id])
        size = (dimensions["x"] * 100, dimensions["y"] * 100, dimensions["z"] * 100)
        position = placement["position"]
        box = {
            "min": [
                position["x"] * 100 - size[0] / 2,
                position["y"] * 100 - size[1] / 2,
                position["z"] * 100 - size[2] / 2,
            ],
            "max": [
                position["x"] * 100 + size[0] / 2,
                position["y"] * 100 + size[1] / 2,
                position["z"] * 100 + size[2] / 2,
            ],
        }
        return box

    def intersect_3d(self, box1, box2):
        # box1 and box2 are dictionaries with 'min' and 'max' keys,
        # which are tuples representing the minimum and maximum corners of the 3D box.
        for i in range(3):
            if box1["max"][i] < box2["min"][i] or box1["min"][i] > box2["max"][i]:
                return False
        return True
