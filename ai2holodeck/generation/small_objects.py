import copy
import itertools
import random
import warnings
from typing import Sequence, Tuple, Dict, List, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from ai2thor.controller import Controller
from ai2thor.hooks.procedural_asset_hook import ProceduralAssetHookRunner
from procthor.constants import FLOOR_Y
from procthor.utils.types import Vector3

from ai2holodeck.constants import THOR_COMMIT_ID, MULTIPROCESSING
from ai2holodeck.generation.holodeck_types import (
    HolodeckScenePlanDict,
    ObjectPlanForSceneDict,
    SmallObjectPlan,
    SelectedSmallObject,
    PlacedSmallObject,
)
from ai2holodeck.generation.llm import OpenAIWithTracking
from ai2holodeck.generation.objaverse_retriever import ObjathorRetriever
from ai2holodeck.generation.object_selector import compare_object_name_to_descriptions
from ai2holodeck.generation.utils import (
    get_bbox_dims,
    get_annotations,
    get_secondary_properties,
    get_bbox_dims_vec,
    get_executor,
    wait_for_futures_and_raise_errors,
)

PROB_SMALL_OBJECT_ROTATION_LARGE = 0.2


class SmallObjectGenerator:
    def __init__(self, object_retriever: ObjathorRetriever, llm: OpenAIWithTracking):
        self.llm = llm
        self.object_retriever = object_retriever
        self.database = object_retriever.database

        # set kinematic to false for small objects
        self.json_template: PlacedSmallObject = {
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

        self.multiprocessing = MULTIPROCESSING

    def select_small_objects_from_plan(
        self,
        scene: HolodeckScenePlanDict,
        controller: Controller,
        receptacle_ids: Sequence[str],
    ) -> Tuple[List[PlacedSmallObject], Dict[str, SelectedSmallObject]]:
        object_selection_plan = scene["object_selection_plan"]

        receptacle_id_to_asset_id = self.get_receptacle_id_to_asset_id(scene=scene)

        if "receptacle_id_to_small_objects" in scene and self.reuse_assets:
            receptacle_id_to_small_objects: Dict[str, SelectedSmallObject] = scene[
                "receptacle_id_to_small_objects"
            ]
        else:
            receptacle_id_to_small_objects: Dict[str, SelectedSmallObject] = (
                self.select_small_object_instances_for_plan(
                    object_selection_plan=object_selection_plan,
                    receptacle_ids=receptacle_ids,
                    receptacle_id_to_asset_id=receptacle_id_to_asset_id,
                )
            )

        results: List[PlacedSmallObject] = []
        # Place the objects
        for receptacle_id, small_objects in receptacle_id_to_small_objects.items():
            placements = []
            for object_name, asset_id, _, _ in small_objects:
                # check if the object is thin and, if so, rotate it so it lies flat
                thin, rotation = self.check_thin_asset(asset_id)

                # check if the object is small and rotate around y axis randomly
                small, y_rotation = self.check_small_asset(asset_id)

                receptacle_obj = next(
                    obj
                    for obj in controller.last_event.metadata["objects"]
                    if obj["objectId"] == receptacle_id
                )
                if rotation[0] == rotation[2] == 0:
                    # Rotate the object to match the receptacle
                    rotation[1] = receptacle_obj["rotation"]["y"] + y_rotation

                obj = self.attempt_to_place_object_on_receptacle_in_scene(
                    controller=controller,
                    asset_id=asset_id,
                    receptacle_id=receptacle_id,
                    rotation=rotation,
                )

                if obj is not None:  # If the object is successfully placed
                    placement: PlacedSmallObject = self.json_template.copy()
                    placement["assetId"] = asset_id
                    placement["id"] = f"{object_name}|{receptacle_id}"
                    placement["position"] = obj["position"]
                    asset_height = get_bbox_dims(self.database[asset_id])["y"]

                    if obj["position"]["y"] + asset_height > scene["wall_height"]:
                        continue  # if the object is too high, skip it

                    placement["position"]["y"] = (
                        obj["position"]["y"] + (asset_height / 2) + 0.001
                    )  # add half of the height to the y position and a small offset
                    placement["rotation"] = obj["rotation"]
                    placement["roomId"] = (
                        receptacle_id.split("(")[1].split(")")[0].strip()
                    )

                    if thin:
                        # TODO: temporary solution fix position and rotation for thin objects
                        placement = self.fix_placement_for_thin_assets(placement)

                    if small:
                        # TODO: temporary solution for random rotation around y axis for small objects
                        placement["rotation"]["y"] = y_rotation

                    if not small and not thin:
                        # set kinematic to true for non-small objects
                        placement["kinematic"] = True

                    if "CanBreak" in get_secondary_properties(self.database[asset_id]):
                        placement["kinematic"] = True

                    placements.append(placement)

            # TODO: check collision between small objects on the same receptacle
            valid_placements = self.check_collision(placements)
            results.extend(valid_placements)

        controller.stop()
        return results, receptacle_id_to_small_objects

    def get_receptacle_id_to_asset_id(self, scene: HolodeckScenePlanDict):
        receptacle_id_to_asset_id = {}
        for object in scene["objects"]:
            receptacle_id_to_asset_id[object["id"]] = object["assetId"]
        return receptacle_id_to_asset_id

    def get_receptacle_id_to_rotation(self, scene: HolodeckScenePlanDict):
        receptacle_id_to_rotation = {}
        for object in scene["objects"]:
            receptacle_id_to_rotation[object["id"]] = object["rotation"]
        return receptacle_id_to_rotation

    def get_receptacle_id_to_position(self, scene: HolodeckScenePlanDict):
        receptacle2rotation = {}
        for object in scene["objects"]:
            receptacle2rotation[object["id"]] = object["position"]
        return receptacle2rotation

    def select_small_object_instances_for_plan(
        self,
        object_selection_plan: ObjectPlanForSceneDict,
        receptacle_ids: Sequence[str],
        receptacle_id_to_asset_id: Sequence[str],
    ):
        child_plans = []
        for room_name, objects in object_selection_plan.items():
            for object_name, object_info in objects.items():
                for child in object_info["objects_on_top"]:
                    child_plan = copy.deepcopy(child)
                    child_plan["room_name"] = room_name
                    child_plan["parent"] = object_name
                    child_plans.append(child_plan)

        receptacle_id_to_small_object_plans = {}
        for receptacle_id in receptacle_ids:
            small_object_plans_for_receptacle = []

            for child_plan in child_plans:
                # TODO: This is a silly way to check if the child is in the receptacle based
                #  on partial string matching parts of the receptacle id
                if (
                    child_plan["room_name"] in receptacle_id
                    and child_plan["parent"] in receptacle_id
                ):
                    small_object_plans_for_receptacle.append(child_plan)

            if len(small_object_plans_for_receptacle) > 0:
                receptacle_id_to_small_object_plans[receptacle_id] = (
                    small_object_plans_for_receptacle
                )

        packed_kwargs = [
            {
                "receptacle_id": receptacle_id,
                "receptacle_asset_id": receptacle_id_to_asset_id[receptacle_id],
                "small_objects": small_objects,
            }
            for receptacle_id, small_objects in receptacle_id_to_small_object_plans.items()
        ]

        with get_executor(self.multiprocessing) as executor:
            return dict(
                wait_for_futures_and_raise_errors(
                    [
                        executor.submit(
                            self.select_small_object_instances_on_receptacle, **kwargs
                        )
                        for kwargs in packed_kwargs
                    ]
                )
            )

    def _select_small_object_instances_on_receptacle(self, kwargs):
        return self.select_small_object_instances_on_receptacle(**kwargs)

    def select_small_object_instances_on_receptacle(
        self,
        receptacle_id: str,
        receptacle_asset_id: str,
        small_objects: Sequence[SmallObjectPlan],
    ) -> Tuple[str, Sequence[SelectedSmallObject]]:
        results = []
        receptacle_dimensions = get_bbox_dims(self.database[receptacle_asset_id])
        receptacle_size = [receptacle_dimensions["x"], receptacle_dimensions["z"]]
        receptacle_area = receptacle_size[0] * receptacle_size[1]
        capacity = 0
        num_objects = 0
        receptacle_size.sort()

        for small_object in sorted(small_objects, key=lambda x: -x["importance"]):
            object_name, quantity, variance_type, importance = (
                small_object["object_name"],
                small_object["quantity"],
                small_object["variance_type"],
                small_object["importance"],
            )
            # maximum 5 of the same object type/name per receptacle
            quantity = min(quantity, 5)
            print(
                f"Placing on {receptacle_id}: selecting {quantity} {object_name} with importance {importance}"
            )
            # Select the object
            candidates: Sequence[Tuple[str, float]] = (
                self.object_retriever.retrieve_with_name_and_desc(
                    object_names=[object_name],
                    object_descriptions=None,
                    threshold=self.clip_threshold,
                )
            )
            candidates = [
                candidate
                for candidate in candidates
                if get_annotations(self.database[candidate[0]])["onObject"]
            ]  # Only select objects that can be placed on other objects

            valid_candidates = []  # Only select objects with high confidence

            for candidate in candidates:
                candidate_dimensions = get_bbox_dims(self.database[candidate[0]])
                candidate_size = [candidate_dimensions["x"], candidate_dimensions["z"]]
                candidate_size.sort()
                if (
                    candidate_size[0] < receptacle_size[0] * 0.9
                    and candidate_size[1] < receptacle_size[1] * 0.9
                ):  # if the object is smaller than the receptacle, threshold is 90%
                    valid_candidates.append(candidate)

            valid_candidates = valid_candidates[:25]
            candidate_scores = compare_object_name_to_descriptions(
                object_name=object_name,
                asset_ids=[candidate[0] for candidate in valid_candidates],
                database=self.database,
                llm=self.llm,
            )
            score_and_val_candidate_list = list(zip(candidate_scores, valid_candidates))
            score_and_val_candidate_list.sort(key=lambda x: x[0], reverse=True)

            valid_candidates = [
                candidate
                for candidate_scores, candidate in score_and_val_candidate_list
                if candidate_scores >= 5
            ]

            if len(valid_candidates) == 0:
                warnings.warn(f"No valid candidate for {object_name}.")
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
            else:
                raise NotImplementedError(
                    f"Variance type {variance_type} not supported."
                )

            for i in range(quantity):
                results.append(
                    (f"{object_name}-{i}", selected_asset_ids[i], importance)
                )

            print(f"Small objects selected for {object_name}: {results[-quantity:]}")

        ordered_small_objects = []
        for object_name, asset_id, importance in results:
            dimensions = get_bbox_dims(self.database[asset_id])
            size = max(dimensions["x"], dimensions["z"])
            ordered_small_objects.append(
                SelectedSmallObject(object_name, asset_id, importance, size)
            )

        ordered_small_objects.sort(key=lambda x: x[-2:], reverse=True)

        return receptacle_id, ordered_small_objects

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

    def attempt_to_place_object_on_receptacle_in_scene(
        self,
        controller: Controller,
        asset_id: str,
        receptacle_id: str,
        rotation=(0, 0, 0),
    ) -> Optional[Dict[str, Any]]:

        generated_id = f"small|{asset_id}|"
        generated_id = generated_id + str(
            sum(
                [
                    obj["objectId"].startswith(generated_id)
                    for obj in controller.last_event.metadata["objects"]
                ],
                0,
            )
        )

        # Spawn the object
        try:
            controller.step(
                action="SpawnAsset",
                assetId=asset_id,
                generatedId=generated_id,
                position=Vector3(x=0, y=FLOOR_Y - 20, z=0),
                rotation=Vector3(x=rotation[0], y=rotation[1], z=rotation[2]),
                renderImage=False,
                raise_for_failure=True,
            )
        except (SystemExit, KeyboardInterrupt):
            raise
        except:
            warnings.warn(
                f"Failed to spawn {asset_id} with {generated_id}: {controller.last_event.metadata['errorMessage']}"
            )
            return None

        def get_obj():
            obj = next(
                obj
                for obj in event.metadata["objects"]
                if obj["objectId"] == generated_id
            )
            return obj

        def object_placed_successfully():
            return (
                receptacle_id
                in controller.step(
                    "CheckWhatObjectOn", objectId=generated_id, belowDistance=2e-2
                ).metadata["actionReturn"]
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
            numPlacementAttempts=100,  # TODO: need to find a better way to determine the number of placement attempts
        )

        if object_placed_successfully():
            print(f"Placed {asset_id} on {receptacle_id} with InitialRandomSpawn")
            return get_obj()
        else:
            receptacle = next(
                obj
                for obj in event.metadata["objects"]
                if obj["objectId"] == receptacle_id
            )
            corners = np.array(receptacle["axisAlignedBoundingBox"]["cornerPoints"])
            min_x, _, min_z = corners.min(0)
            max_x, max_y, max_z = corners.max(0)

            obj = get_obj()
            obj_diameter = max(
                obj["axisAlignedBoundingBox"]["size"]["x"],
                obj["axisAlignedBoundingBox"]["size"]["z"],
            )

            if obj_diameter <= max_x - min_x and obj_diameter <= max_z - min_z:
                random_positions = random.sample(
                    list(
                        itertools.product(
                            np.linspace(min_x + obj_diameter, max_x - obj_diameter, 10),
                            [max_y + 0.01],
                            np.linspace(min_z + obj_diameter, max_z - obj_diameter, 10),
                        )
                    ),
                    10,
                )

                for x, y, z in random_positions:
                    event = controller.step(
                        "PlaceObjectAtPoint",
                        position=Vector3(x=x, y=y, z=z),
                        objectId=generated_id,
                    )

                    if not event:
                        continue

                    if object_placed_successfully():
                        print(f"Placed {asset_id} on {receptacle_id} with PlaceAtPoint")
                        break

        if object_placed_successfully():
            return get_obj()
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

    def check_small_asset(self, asset_id: str):
        dims = get_bbox_dims_vec(self.database[asset_id])
        if (dims < 0.25).all():
            if random.random() < PROB_SMALL_OBJECT_ROTATION_LARGE:
                return True, random.randint(0, 360)
            else:
                return True, random.randint(-15, 15)
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

    def check_collision(self, placements: Sequence[PlacedSmallObject]):
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
