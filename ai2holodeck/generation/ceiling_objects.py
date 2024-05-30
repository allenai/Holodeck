import copy
import re

import torch
import torch.nn.functional as F
from colorama import Fore
from langchain import PromptTemplate, OpenAI
from shapely.geometry import Polygon

import ai2holodeck.generation.prompts as prompts
from ai2holodeck.generation.objaverse_retriever import ObjathorRetriever
from ai2holodeck.generation.utils import get_bbox_dims, get_annotations


class CeilingObjectGenerator:
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
        self.ceiling_template = PromptTemplate(
            input_variables=["input", "rooms", "additional_requirements"],
            template=prompts.ceiling_selection_prompt,
        )

    def generate_ceiling_objects(self, scene, additional_requirements_ceiling="N/A"):
        room_types = [room["roomType"] for room in scene["rooms"]]
        room_types_str = str(room_types).replace("'", "")[1:-1]
        ceiling_prompt = self.ceiling_template.format(
            input=scene["query"],
            rooms=room_types_str,
            additional_requirements=additional_requirements_ceiling,
        )

        if "raw_ceiling_plan" not in scene:
            raw_ceiling_plan = self.llm(ceiling_prompt)
        else:
            raw_ceiling_plan = scene["raw_ceiling_plan"]

        print(f"\nUser: {ceiling_prompt}\n")
        print(
            f"{Fore.GREEN}AI: Here is the ceiling plan:\n{raw_ceiling_plan}{Fore.RESET}"
        )

        ceiling_objects = []
        parsed_ceiling_plan = self.parse_ceiling_plan(raw_ceiling_plan)
        for room_type, ceiling_object_description in parsed_ceiling_plan.items():
            room = self.get_room_by_type(scene["rooms"], room_type)

            if room is None:
                print(f"Room type {room_type} not found in scene.")
                continue

            ceiling_object_id = self.select_ceiling_object(ceiling_object_description)
            if ceiling_object_id is None:
                continue

            # Temporary solution: place at the center of the room
            dimension = get_bbox_dims(self.database[ceiling_object_id])

            floor_polygon = Polygon(room["vertices"])
            x = floor_polygon.centroid.x
            z = floor_polygon.centroid.y
            y = scene["wall_height"] - dimension["y"] / 2

            ceiling_object = copy.deepcopy(self.json_template)
            ceiling_object["assetId"] = ceiling_object_id
            ceiling_object["id"] = f"ceiling ({room_type})"
            ceiling_object["position"] = {"x": x, "y": y, "z": z}
            ceiling_object["rotation"] = {"x": 0, "y": 0, "z": 0}
            ceiling_object["roomId"] = room["id"]
            ceiling_object["object_name"] = get_annotations(
                self.database[ceiling_object_id]
            )["category"]
            ceiling_objects.append(ceiling_object)

        return raw_ceiling_plan, ceiling_objects

    def parse_ceiling_plan(self, raw_ceiling_plan):
        plans = [plan.lower() for plan in raw_ceiling_plan.split("\n") if "|" in plan]
        parsed_plans = {}
        for plan in plans:
            # remove index
            pattern = re.compile(r"^\d+\.\s*")
            plan = pattern.sub("", plan)
            if plan[-1] == ".":
                plan = plan[:-1]  # remove the last period

            room_type, ceiling_object_description = plan.split("|")
            room_type = room_type.strip()
            ceiling_object_description = ceiling_object_description.strip()
            if (
                room_type not in parsed_plans
            ):  # only consider one type of ceiling object for each room
                parsed_plans[room_type] = ceiling_object_description
        return parsed_plans

    def get_room_by_type(self, rooms, room_type):
        for room in rooms:
            if room["roomType"] == room_type:
                return room
        return None

    def select_ceiling_object(self, description):
        candidates = self.object_retriever.retrieve(
            [f"a 3D model of {description}"], threshold=29
        )
        ceiling_candiates = [
            candidate
            for candidate in candidates
            if get_annotations(self.database[candidate[0]])["onCeiling"] == True
        ]

        valid_ceiling_candiates = []
        for candidate in ceiling_candiates:
            dimension = get_bbox_dims(self.database[candidate[0]])
            if dimension["y"] <= 1.0:
                valid_ceiling_candiates.append(candidate)

        if len(valid_ceiling_candiates) == 0:
            print("No ceiling object found for description: {}".format(description))
            return None

        selected_ceiling_object_id = self.random_select(valid_ceiling_candiates)[0]
        return selected_ceiling_object_id

    def random_select(self, candidates):
        scores = [candidate[1] for candidate in candidates]
        scores_tensor = torch.Tensor(scores)
        probas = F.softmax(
            scores_tensor, dim=0
        )  # TODO: consider using normalized scores
        selected_index = torch.multinomial(probas, 1).item()
        selected_candidate = candidates[selected_index]
        return selected_candidate
