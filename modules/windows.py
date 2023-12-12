import re
import ast
import copy
import json
import random
import numpy as np
from colorama import Fore
import modules.prompts as prompts
from langchain import PromptTemplate

class WindowGenerator():
    def __init__(self, llm):
        self.json_template = {"assetId": None, "id": None, "room0": None, "room1": None,
                              "wall0": None, "wall1": None, "holePolygon": [],
                              "assetPosition": {}, "roomId": None}
        
        self.window_data = json.load(open("data/windows/window-database.json", "r"))
        self.window_ids = list(self.window_data.keys())
        self.hole_offset = 0.05 # make the hole smaller than windows
        self.llm = llm
        self.window_template = PromptTemplate(input_variables=["input", "walls", "wall_height", "additional_requirements"],
                                              template=prompts.window_prompt)
        self.used_assets = []

    
    def generate_windows(self, scene, additional_requirements_window):
        # get organized walls
        organized_walls, available_wall_str = self.get_wall_for_windows(scene)
        window_prompt = self.window_template.format(input=scene["query"],
                                                    walls=available_wall_str,
                                                    wall_height=int(scene["wall_height"] * 100),
                                                    additional_requirements=additional_requirements_window)
        
        if "raw_window_plan" not in scene: raw_window_plan = self.llm(window_prompt)
        else: raw_window_plan = scene["raw_window_plan"]

        print(f"\nUser: {window_prompt}\n")
        print(f"{Fore.GREEN}AI: Here is the window plan:\n{raw_window_plan}{Fore.RESET}")

        walls = scene["walls"]
        windows = []
        window_ids = []
        rows = [row.lower() for row in raw_window_plan.split("\n") if "|" in row]
        room_with_windows = []
        for row in rows:
            # parse window plan
            parsed_plan = self.parse_raw_plan(row)
            if parsed_plan is None: continue
            
            # get room id
            room_id = parsed_plan["room_id"]

            # only one wall with windows per room
            if room_id not in room_with_windows: room_with_windows.append(room_id)
            else: print(f"Warning: room {room_id} already has windows"); continue

            # get wall id
            try: wall_id = organized_walls[room_id][parsed_plan["wall_direction"]]["wall_id"]
            except: print("Warning: no available wall for {}".format(row)); continue
            
            for wall in walls:
                if wall["id"] == wall_id:
                    wall_info = wall

            # select window
            window_id = self.select_window(parsed_plan["window_type"], parsed_plan["window_size"])
            window_polygons, window_positions, window_segments, window_boxes, new_wall_ids, updated_walls = self.get_window_polygon(window_id, parsed_plan["window_height"], parsed_plan["quantity"], wall_info, walls)
            walls = updated_walls # update walls

            if window_polygons == []: print("Warning: no windows generated for {}".format(row)); continue

            # generate window json
            for i in range(len(window_polygons)):
                current_wall_id = new_wall_ids[i]
                current_window = copy.deepcopy(self.json_template)
                current_window["wall0"] = current_wall_id
                current_window["wall1"] = current_wall_id + "|exterior"
                current_window["room0"] = room_id
                current_window["room1"] = room_id
                current_window["roomId"] = room_id
                current_window["assetId"] = window_id
                current_window["id"] = f"window|{current_wall_id}|{i}"
                current_window["holePolygon"] = window_polygons[i]
                current_window["assetPosition"] = window_positions[i]
                current_window["windowSegment"] = window_segments[i]
                current_window["windowBoxes"] = window_boxes[i]

                # sometimes the same window is generated twice and causes errors
                if current_window["id"] not in window_ids:
                    window_ids.append(current_window["id"])
                    windows.append(current_window)
                else:
                    print("Warning: duplicated window id: {}".format(current_window["id"]))

        return raw_window_plan, walls, windows

            
    def parse_raw_plan(self, plan):
        try:
            pattern = re.compile(r'^(\d+[\.\)]\s*|- )')
            plan = pattern.sub('', plan)
            if plan[-1] == ".": plan = plan[:-1] # remove the last period
            room_id, wall_direction, window_type, window_size, quantity, window_height = plan.split("|")
            return {"room_id": room_id.strip(),
                    "wall_direction": wall_direction.strip().lower(),
                    "window_type": window_type.strip().lower(),
                    "window_size": ast.literal_eval(window_size.strip()),
                    "quantity": int(quantity.strip()),
                    "window_height": float(window_height.strip())}
        except:
            print("Error: could not parse window plan: {}".format(plan))
            return None
    

    def get_room(self, rooms, room_type):
        for room in rooms:
            if room_type == room["roomType"]: return room
    

    def get_wall_for_windows(self, scene):
        walls_with_door = []
        for door in scene["doors"]:
            walls_with_door.append(door["wall0"])
            walls_with_door.append(door["wall1"])

        available_walls = []

        for wall in scene["walls"]:
            if "connect_exterior" in wall and wall["id"] not in walls_with_door:
                available_walls.append(wall)

        organized_walls = {}
        for wall in available_walls:
            room_id = wall['roomId']
            wall_direction = wall['direction']

            wall_width = wall['width']
            if wall_width < 2.0: continue

            if room_id not in organized_walls: organized_walls[room_id] = {}
            
            if wall_direction not in organized_walls[room_id]:
                organized_walls[room_id][wall_direction] = {"wall_id": wall["id"], "wall_width": wall_width}
            else:
                if wall_width > organized_walls[room_id][wall_direction]["wall_width"]:
                    organized_walls[room_id][wall_direction] = {"wall_id": wall["id"], "wall_width": wall_width}
        
        available_wall_str = ""
        for room_id in organized_walls:
            current_str = "{}: ".format(room_id)
            for wall_direction in organized_walls[room_id]:
                current_str += "{}, {} cm; ".format(wall_direction, int(organized_walls[room_id][wall_direction]["wall_width"] * 100))
            available_wall_str += current_str + "\n"

        return organized_walls, available_wall_str
    

    def select_window(self, window_type, window_size):
        candidate_window_ids = [window_id for window_id in self.window_ids if self.window_data[window_id]["type"] == window_type]
        size_differences = [np.linalg.norm(np.array(window_size) - np.array(self.window_data[window_id]["size"])) for window_id in candidate_window_ids]
        sorted_window_ids = [x for _, x in sorted(zip(size_differences, candidate_window_ids))]

        top_window_ids = sorted_window_ids[0]
        sorted_window_ids = [window_id for window_id in sorted_window_ids if window_id not in self.used_assets]

        if len(sorted_window_ids) == 0: selected_window_id = top_window_ids
        else: selected_window_id = sorted_window_ids[0]

        return selected_window_id
    

    def get_window_polygon(self, window_id, window_height, quantity, wall_info, walls):
        window_x = self.window_data[window_id]["boundingBox"]["x"] - self.hole_offset
        window_y = self.window_data[window_id]["boundingBox"]["y"] - self.hole_offset
    
        wall_width = wall_info["width"]
        wall_height = wall_info["height"]
        wall_segment = wall_info["segment"]

        window_height = min(window_height / 100.0, wall_height - window_y)

        quantity = min(quantity, int(wall_width / window_x))

        wall_start = np.array(wall_segment[0])
        wall_end = np.array(wall_segment[1])
        original_vector = wall_end - wall_start
        original_length = np.linalg.norm(original_vector)
        normalized_vector = original_vector / original_length
        subwall_length = original_length / quantity

        if quantity == 0: 
            return [], [], [], [], [], walls

        elif quantity == 1:
            window_start = random.uniform(0, wall_width - window_x)
            window_end = window_start + window_x
            polygon = [{"x": window_start, "y": window_height, "z": 0},
                       {"x": window_end, "y": window_height + window_y, "z": 0}]
            position = {"x": (polygon[0]["x"] + polygon[1]["x"]) / 2,
                        "y": (polygon[0]["y"] + polygon[1]["y"]) / 2,
                        "z": (polygon[0]["z"] + polygon[1]["z"]) / 2}
            window_segment = [list(wall_start + normalized_vector * window_start), list(wall_start + normalized_vector * window_end)]
            window_boxes = self.create_rectangles(window_segment)

            return [polygon], [position], [window_segment], [window_boxes], [wall_info["id"]], walls
    
        else:
            # split walls into subwalls
            segments = []
            for i in range(quantity):
                segment_start = wall_start + i * subwall_length * normalized_vector
                segment_end = wall_start + (i+1) * subwall_length * normalized_vector
                segments.append((segment_start, segment_end))
            
            # update walls
            updated_walls = []
            new_wall_ids = []
            for wall in walls:
                if wall_info["id"] not in wall["id"]:
                    updated_walls.append(wall)

            for i in range(len(segments)):
                # generate new subwall json
                current_wall = copy.deepcopy(wall_info)
                current_wall["id"] = f"{wall_info['id']}|{i}"
                current_wall["segment"] = [segments[i][0].tolist(), segments[i][1].tolist()]
                current_wall["width"] = subwall_length
                current_wall["polygon"] = self.generate_wall_polygon(segments[i][0].tolist(), segments[i][1].tolist(), wall_height)
                current_wall["connect_exterior"] = current_wall["id"] + "|exterior"

                # add exterior wall
                current_wall_exterior = copy.deepcopy(current_wall)
                current_wall_exterior["id"] = current_wall["id"] + "|exterior"
                current_wall_exterior["material"] = {"name": "Walldrywall4Tiled"}
                current_wall_exterior["polygon"] = current_wall["polygon"][::-1]
                current_wall_exterior["segment"] = current_wall["segment"][::-1]
                current_wall_exterior.pop("connect_exterior")

                updated_walls.append(current_wall)
                updated_walls.append(current_wall_exterior)
                new_wall_ids.append(current_wall["id"])
            
            # generate window polygons
            window_polygons = []
            window_positions = []
            window_segments = []
            window_boxes = []
            for i in range(len(segments)):
                window_start = random.uniform(0, subwall_length - window_x) # TODO: consider the same start point for all windows
                window_end = window_start + window_x
                polygon = [{"x": window_start, "y": window_height, "z": 0},
                           {"x": window_end, "y": window_height + window_y, "z": 0}]
                position = {"x": (polygon[0]["x"] + polygon[1]["x"]) / 2,
                            "y": (polygon[0]["y"] + polygon[1]["y"]) / 2,
                            "z": (polygon[0]["z"] + polygon[1]["z"]) / 2}
                
                window_segment = [list(segments[i][0] + normalized_vector * window_start), list(segments[i][0] + normalized_vector * window_end)]
                window_box = self.create_rectangles(window_segment)
                window_polygons.append(polygon)
                window_positions.append(position)
                window_segments.append(window_segment)
                window_boxes.append(window_box)
            
            return window_polygons, window_positions, window_segments, window_boxes, new_wall_ids, updated_walls


    def generate_wall_polygon(self, point, next_point, wall_height):
        wall_polygon = []
        # add the base point
        wall_polygon.append({'x': point[0], 'y': 0, 'z': point[1]})
        # add the top point (with the same x and z, but y = wall_height)
        wall_polygon.append({'x': point[0], 'y': wall_height, 'z': point[1]})
        # add the top point of the next base point
        wall_polygon.append({'x': next_point[0], 'y': wall_height, 'z': next_point[1]})
        # add the next base point
        wall_polygon.append({'x': next_point[0], 'y': 0, 'z': next_point[1]})
        return wall_polygon
    

    def create_rectangles(self, segment):
        # Convert to numpy arrays for easier calculations
        pt1 = np.array(segment[0])
        pt2 = np.array(segment[1])

        # Calculate the vector for the segment
        vec = pt2 - pt1

        # Calculate a perpendicular vector with length 1
        perp_vec = np.array([-vec[1], vec[0]])
        perp_vec /= np.linalg.norm(perp_vec)
        perp_vec *= 0.1 # 0.1 is the width of the window

        # Calculate the four points for each rectangle
        top_rectangle = [list(pt1 + perp_vec), list(pt2 + perp_vec), list(pt2), list(pt1)]
        bottom_rectangle = [list(pt1), list(pt2), list(pt2 - perp_vec), list(pt1 - perp_vec)]

        return top_rectangle, bottom_rectangle