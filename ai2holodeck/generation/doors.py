import copy
import os
import random

import compress_json
import compress_pickle
import numpy as np
import torch
from PIL import Image
from colorama import Fore
from langchain import PromptTemplate
from tqdm import tqdm

import ai2holodeck.generation.prompts as prompts
from ai2holodeck.constants import HOLODECK_BASE_DATA_DIR
from ai2holodeck.generation.llm import OpenAIWithTracking


class DoorGenerator:
    def __init__(
        self, clip_model, clip_preprocess, clip_tokenizer, llm: OpenAIWithTracking
    ):
        self.json_template = {
            "assetId": None,
            "id": None,
            "openable": False,
            "openness": 0,
            "room0": None,
            "room1": None,
            "wall0": None,
            "wall1": None,
            "holePolygon": [],
            "assetPosition": {},
        }

        self.door_data = compress_json.load(
            os.path.join(HOLODECK_BASE_DATA_DIR, "doors/door-database.json")
        )
        self.door_ids = list(self.door_data.keys())
        self.used_assets = []

        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.clip_tokenizer = clip_tokenizer

        self.load_features()
        self.llm = llm
        self.doorway_template = PromptTemplate(
            input_variables=[
                "input",
                "rooms",
                "room_sizes",
                "room_pairs",
                "additional_requirements",
            ],
            template=prompts.DOORWAY_PROMPT,
        )

    def load_features(self):
        try:
            self.door_feature_clip = compress_pickle.load(
                os.path.join(HOLODECK_BASE_DATA_DIR, "doors/door_feature_clip.pkl")
            )
        except:
            print("Precompute image features for doors...")
            self.door_feature_clip = []
            for door_id in tqdm(self.door_ids):
                image = self.preprocess(
                    Image.open(
                        os.path.join(
                            HOLODECK_BASE_DATA_DIR, f"doors/images/{door_id}.png"
                        )
                    )
                ).unsqueeze(0)
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(image)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                self.door_feature_clip.append(image_features)
            self.door_feature_clip = torch.vstack(self.door_feature_clip)
            compress_pickle.dump(
                self.door_feature_clip,
                os.path.join(HOLODECK_BASE_DATA_DIR, "doors/door_feature_clip.pkl"),
            )

    def generate_doors(self, scene, additional_requirements_door):
        # get room pairs
        room_types = [room["roomType"] for room in scene["rooms"]]
        room_types_str = str(room_types).replace("'", "")[1:-1]
        room_pairs = self.get_room_pairs_str(rooms=scene["rooms"], walls=scene["walls"])
        room_sizes_str = self.get_room_size_str(scene)
        room_pairs_str = str(room_pairs).replace("'", "")[1:-1]

        doorway_prompt = self.doorway_template.format(
            input=scene["query"],
            rooms=room_types_str,
            room_sizes=room_sizes_str,
            room_pairs=room_pairs_str,
            additional_requirements=additional_requirements_door,
        )

        # generate raw doorway plan if not exist
        if "raw_doorway_plan" not in scene:
            raw_doorway_plan = self.llm(doorway_prompt)
        else:
            raw_doorway_plan = scene["raw_doorway_plan"]

        print(f"\nUser: {doorway_prompt}\n")
        print(
            f"{Fore.GREEN}AI: Here is the doorway plan:\n{raw_doorway_plan}{Fore.RESET}"
        )

        rooms = scene["rooms"]
        walls = scene["walls"]
        doors = []
        open_room_pairs = []
        plans = [plan.lower() for plan in raw_doorway_plan.split("\n") if "|" in plan]
        room_types = [room["roomType"] for room in rooms] + ["exterior"]
        for i, plan in enumerate(plans):
            # TODO: rewrite the parsing logic
            current_door = copy.deepcopy(self.json_template)
            parsed_plan = self.parse_door_plan(plan)

            if parsed_plan == None:
                continue

            if (
                parsed_plan["room_type0"] not in room_types
                or parsed_plan["room_type1"] not in room_types
            ):
                print(
                    f"{Fore.RED}{parsed_plan['room_type0']} or {parsed_plan['room_type1']} not exist{Fore.RESET}"
                )
                continue

            current_door["room0"] = parsed_plan["room_type0"]
            current_door["room1"] = parsed_plan["room_type1"]
            current_door["id"] = (
                f"door|{i}|{parsed_plan['room_type0']}|{parsed_plan['room_type1']}"
            )

            if parsed_plan["connection_type"] == "open":
                open_room_pairs.append(
                    (parsed_plan["room_type0"], parsed_plan["room_type1"])
                )
                continue

            # get connection
            exterior = False
            if (
                parsed_plan["room_type0"] == "exterior"
                or parsed_plan["room_type1"] == "exterior"
            ):
                connection = self.get_connection_exterior(
                    parsed_plan["room_type0"], parsed_plan["room_type1"], walls
                )
                exterior = True
            else:
                connection = self.get_connection(
                    parsed_plan["room_type0"], parsed_plan["room_type1"], walls
                )

            if connection == None:
                continue

            # get wall information
            current_door["wall0"] = connection["wall0"]
            current_door["wall1"] = connection["wall1"]

            # get door asset
            if exterior:
                parsed_plan["connection_type"] = (
                    "doorway"  # force to use doorway for exterior
                )
            door_id = self.select_door(
                parsed_plan["connection_type"],
                parsed_plan["size"],
                parsed_plan["style"],
            )
            current_door["assetId"] = door_id

            if parsed_plan["connection_type"] == "doorway" and not exterior:
                current_door["openable"] = True
                current_door["openness"] = 1

            # get polygon
            door_dimension = self.door_data[door_id]["boundingBox"]
            door_polygon = self.get_door_polygon(
                connection["segment"], door_dimension, parsed_plan["connection_type"]
            )

            if door_polygon != None:
                polygon, position, door_boxes, door_segment = door_polygon
                current_door["holePolygon"] = polygon
                current_door["assetPosition"] = position
                current_door["doorBoxes"] = door_boxes
                current_door["doorSegment"] = door_segment
                doors.append(current_door)

        # check if there is any room has no door
        connected_rooms = []
        for door in doors:
            connected_rooms.append(door["room0"])
            connected_rooms.append(door["room1"])

        for pair in open_room_pairs:
            connected_rooms.append(pair[0])
            connected_rooms.append(pair[1])

        unconnected_rooms = []
        for room in rooms:
            if room["roomType"] not in connected_rooms:
                unconnected_rooms.append(room["roomType"])

        if len(unconnected_rooms) > 0:
            for room in unconnected_rooms:
                if room in connected_rooms:
                    continue

                current_door = copy.deepcopy(self.json_template)
                current_walls = [
                    wall
                    for wall in walls
                    if wall["roomId"] == room
                    and "exterior" not in wall["id"]
                    and len(wall["connected_rooms"]) != 0
                ]
                widest_wall = max(current_walls, key=lambda x: x["width"])

                room_to_connect = widest_wall["connected_rooms"][0]["roomId"]
                current_door["room0"] = room
                current_door["room1"] = room_to_connect

                current_door["id"] = f"door|{i}|{room}|{room_to_connect}"

                wall_to_connect = widest_wall["connected_rooms"][0]["wallId"]
                current_door["wall0"] = widest_wall["id"]
                current_door["wall1"] = wall_to_connect

                # get door asset
                door_id = self.get_random_door(widest_wall["width"])
                current_door["assetId"] = door_id

                # get polygon
                door_dimension = self.door_data[door_id]["boundingBox"]
                door_type = self.door_data[door_id]["type"]

                door_polygon = self.get_door_polygon(
                    widest_wall["connected_rooms"][0]["intersection"],
                    door_dimension,
                    door_type,
                )

                if door_polygon != None:
                    polygon, position, door_boxes, door_segment = door_polygon
                    current_door["holePolygon"] = polygon
                    current_door["assetPosition"] = position
                    current_door["doorBoxes"] = door_boxes
                    current_door["doorSegment"] = door_segment
                    doors.append(current_door)

                    connected_rooms.append(room)
                    connected_rooms.append(room_to_connect)

        return raw_doorway_plan, doors, room_pairs, open_room_pairs

    def get_room(self, rooms, room_type):
        for room in rooms:
            if room_type == room["roomType"]:
                return room

    def parse_door_plan(self, plan):
        try:
            room_type0, room_type1, connection_type, size, style = plan.split("|")
            return {
                "room_type0": room_type0.strip(),
                "room_type1": room_type1.strip(),
                "connection_type": connection_type.strip(),
                "size": size.strip(),
                "style": style.strip(),
            }
        except:
            print(f"{Fore.RED}Invalid door plan:{Fore.RESET}", plan)
            return None

    def get_door_polygon(self, segment, door_dimension, connection_type):
        door_width = door_dimension["x"]
        door_height = door_dimension["y"]

        start = np.array([segment[0]["x"], segment[0]["z"]])
        end = np.array([segment[1]["x"], segment[1]["z"]])

        original_vector = end - start
        original_length = np.linalg.norm(original_vector)
        normalized_vector = original_vector / original_length

        if door_width >= original_length:
            print(f"{Fore.RED}The wall is too narrow to install a door.{Fore.RESET}")
            return None

        else:
            door_start = random.uniform(0, original_length - door_width)
            door_end = door_start + door_width

            polygon = [
                {"x": door_start, "y": 0, "z": 0},
                {"x": door_end, "y": door_height, "z": 0},
            ]

            door_segment = [
                list(start + normalized_vector * door_start),
                list(start + normalized_vector * door_end),
            ]
            door_boxes = self.create_rectangles(door_segment, connection_type)

            position = {
                "x": (polygon[0]["x"] + polygon[1]["x"]) / 2,
                "y": (polygon[0]["y"] + polygon[1]["y"]) / 2,
                "z": (polygon[0]["z"] + polygon[1]["z"]) / 2,
            }

            return polygon, position, door_boxes, door_segment

    def get_connection(self, room0_id, room1_id, walls):
        room0_walls = [wall for wall in walls if wall["roomId"] == room0_id]
        valid_connections = []
        for wall in room0_walls:
            connections = wall["connected_rooms"]
            if len(connections) != 0:
                for connection in connections:
                    if connection["roomId"] == room1_id:
                        valid_connections.append(
                            {
                                "wall0": wall["id"],
                                "wall1": connection["wallId"],
                                "segment": connection["intersection"],
                            }
                        )

        if len(valid_connections) == 0:
            print(
                f"{Fore.RED}There is no wall between {room0_id} and {room1_id}{Fore.RESET}"
            )
            return None

        elif len(valid_connections) == 1:
            connection = valid_connections[0]

        else:  # handle the case when there are multiple ways
            print(
                f"{Fore.RED}There are multiple ways between {room0_id} and {room1_id}{Fore.RESET}"
            )
            longest_segment_length = 0
            connection = None
            for current_connection in valid_connections:
                current_segment = current_connection["segment"]
                current_segment_length = np.linalg.norm(
                    np.array([current_segment[0]["x"], current_segment[0]["z"]])
                    - np.array([current_segment[1]["x"], current_segment[1]["z"]])
                )
                if current_segment_length > longest_segment_length:
                    connection = current_connection
                    longest_segment_length = current_segment_length

        return connection

    def get_connection_exterior(self, room0_id, room1_id, walls):
        room_id = room0_id if room0_id != "exterior" else room1_id
        interior_walls = [
            wall["id"]
            for wall in walls
            if wall["roomId"] == room_id and "exterior" not in wall["id"]
        ]
        exterior_walls = [
            wall["id"]
            for wall in walls
            if wall["roomId"] == room_id and "exterior" in wall["id"]
        ]
        wall_pairs = []
        for interior_wall in interior_walls:
            for exterior_wall in exterior_walls:
                if interior_wall in exterior_wall:
                    wall_pairs.append({"wall0": exterior_wall, "wall1": interior_wall})

        valid_connections = []
        for wall_pair in wall_pairs:
            wall0 = wall_pair["wall0"]
            wall1 = wall_pair["wall1"]
            for wall in walls:
                if wall["id"] == wall0:
                    wall0_segment = wall["segment"]
                    break
            segment = [
                {"x": wall0_segment[0][0], "y": 0.0, "z": wall0_segment[0][1]},
                {"x": wall0_segment[1][0], "y": 0.0, "z": wall0_segment[1][1]},
            ]

            valid_connections.append(
                {"wall0": wall0, "wall1": wall1, "segment": segment}
            )

        if len(valid_connections) == 0:
            return None

        elif len(valid_connections) == 1:
            return valid_connections[0]

        else:
            print(
                f"{Fore.RED}There are multiple ways between {room0_id} and {room1_id}{Fore.RESET}"
            )
            longest_segment_length = 0
            connection = None
            for current_connection in valid_connections:
                current_segment = current_connection["segment"]
                current_segment_length = np.linalg.norm(
                    np.array([current_segment[0]["x"], current_segment[0]["z"]])
                    - np.array([current_segment[1]["x"], current_segment[1]["z"]])
                )
                if current_segment_length > longest_segment_length:
                    connection = current_connection
                    longest_segment_length = current_segment_length

            return connection

    def select_door(self, door_type, door_size, query):
        with torch.no_grad():
            query_feature_clip = self.clip_model.encode_text(
                self.clip_tokenizer([query])
            )
            query_feature_clip /= query_feature_clip.norm(dim=-1, keepdim=True)

        clip_similarity = query_feature_clip @ self.door_feature_clip.T
        sorted_indices = torch.argsort(clip_similarity, descending=True)[0]
        valid_door_ids = []
        for ind in sorted_indices:
            door_id = self.door_ids[ind]
            if (
                self.door_data[door_id]["type"] == door_type
                and self.door_data[door_id]["size"] == door_size
            ):
                valid_door_ids.append(door_id)

        top_door_id = valid_door_ids[0]
        valid_door_ids = [
            door_id for door_id in valid_door_ids if door_id not in self.used_assets
        ]
        if len(valid_door_ids) == 0:
            valid_door_ids = [top_door_id]

        return valid_door_ids[0]

    def create_rectangles(self, segment, connection_type):
        box_width = 1.0
        if connection_type == "doorframe":
            box_width = 1.0

        # Convert to numpy arrays for easier calculations
        pt1 = np.array(segment[0])
        pt2 = np.array(segment[1])

        # Calculate the vector for the segment
        vec = pt2 - pt1

        # Calculate a perpendicular vector with length 1
        perp_vec = np.array([-vec[1], vec[0]])
        perp_vec /= np.linalg.norm(perp_vec)
        perp_vec *= box_width

        # Calculate the four points for each rectangle
        top_rectangle = [
            list(pt1 + perp_vec),
            list(pt2 + perp_vec),
            list(pt2),
            list(pt1),
        ]
        bottom_rectangle = [
            list(pt1),
            list(pt2),
            list(pt2 - perp_vec),
            list(pt1 - perp_vec),
        ]

        return top_rectangle, bottom_rectangle

    def get_room_pairs_str(self, rooms, walls):
        room_pairs = [
            (wall["roomId"], wall["connected_rooms"][0]["roomId"])
            for wall in walls
            if len(wall["connected_rooms"]) == 1 and wall["width"] >= 2.0
        ]
        for wall in walls:
            if "exterior" in wall["id"]:
                room_pairs.append(("exterior", wall["roomId"]))

        room_pairs_no_dup = []
        for pair in room_pairs:
            if (
                pair not in room_pairs_no_dup
                and (pair[1], pair[0]) not in room_pairs_no_dup
            ):
                room_pairs_no_dup.append(pair)

        room_pairs_clean = []
        existed_rooms = []
        for pair in room_pairs_no_dup:
            if pair[0] not in existed_rooms or pair[1] not in existed_rooms:
                room_pairs_clean.append(pair)

            if pair[0] not in existed_rooms:
                existed_rooms.append(pair[0])
            if pair[1] not in existed_rooms:
                existed_rooms.append(pair[1])

        return room_pairs_clean

    def get_room_size_str(self, scene):
        wall_height = scene["wall_height"]
        room_size_str = ""
        for room in scene["rooms"]:
            room_name = room["roomType"]
            room_size = self.get_room_size(room)
            room_size_str += (
                f"{room_name}: {room_size[0]} m x {room_size[1]} m x {wall_height} m\n"
            )

        return room_size_str

    def get_room_size(self, room):
        floor_polygon = room["floorPolygon"]
        x_values = [point["x"] for point in floor_polygon]
        z_values = [point["z"] for point in floor_polygon]
        return (max(x_values) - min(x_values), max(z_values) - min(z_values))

    def get_random_door(self, wall_width):
        single_doors = [
            door_id
            for door_id in self.door_ids
            if self.door_data[door_id]["size"] == "single"
        ]
        double_doors = [
            door_id
            for door_id in self.door_ids
            if self.door_data[door_id]["size"] == "double"
        ]

        if wall_width < 2.0:
            return random.choice(single_doors)
        else:
            return random.choice(double_doors + single_doors)
