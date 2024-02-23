import copy
import random

import numpy as np
from colorama import Fore
from langchain import PromptTemplate, OpenAI
from shapely.geometry import LineString, Polygon, Point

import holodeck.generation.prompts as prompts


class WallGenerator:
    def __init__(self, llm: OpenAI):
        self.json_template = {
            "id": None,
            "roomId": None,
            "material": {"name": None, "color": None},
            "polygon": [],
        }
        self.llm = llm
        self.wall_height_template = PromptTemplate(
            input_variables=["input"], template=prompts.wall_height_prompt
        )
        self.used_assets = []

    def generate_walls(self, scene):
        # get wall height
        if "wall_height" not in scene:
            wall_height = self.get_wall_height(scene)
        else:
            wall_height = scene["wall_height"]

        walls = []
        rooms = scene["rooms"]
        for room in rooms:
            roomId = room["id"]
            material = room["wallMaterial"]
            full_vertices = room["full_vertices"]

            for j in range(len(full_vertices)):
                wall = copy.deepcopy(self.json_template)
                wall["roomId"] = roomId
                wall["material"] = material

                # generate the wall polygon
                wall_endpoint1 = full_vertices[j]
                wall_endpoint2 = full_vertices[(j + 1) % len(full_vertices)]
                wall["polygon"] = self.generate_wall_polygon(
                    wall_endpoint1, wall_endpoint2, wall_height
                )

                # add the room connection information
                wall["connected_rooms"] = self.get_connected_rooms(
                    wall["polygon"], rooms, wall["roomId"]
                )

                # add the wall direction and width
                wall_width, wall_direction = self.get_wall_direction(
                    wall_endpoint1, wall_endpoint2, full_vertices
                )
                wall["width"] = wall_width
                wall["height"] = wall_height
                wall["direction"] = wall_direction
                wall["segment"] = [wall_endpoint1, wall_endpoint2]
                wall["id"] = f"wall|{roomId}|{wall_direction}|{j}"
                walls.append(wall)

        # update wall connection information
        for wall in walls:
            if wall["connected_rooms"] != []:
                for connection in wall["connected_rooms"]:
                    connect_room_id = connection["roomId"]
                    candidate_walls = [
                        wall for wall in walls if wall["roomId"] == connect_room_id
                    ]
                    line1 = connection["line1"]
                    for candidate_wall in candidate_walls:
                        if (
                            line1[0] in candidate_wall["polygon"]
                            and line1[1] in candidate_wall["polygon"]
                        ):
                            connection["wallId"] = candidate_wall["id"]

        # add exterior walls
        updated_walls = []
        for wall in walls:
            if wall["connected_rooms"] == []:
                exterior_wall = copy.deepcopy(wall)
                exterior_wall["id"] = wall["id"] + "|exterior"
                exterior_wall["material"] = {"name": "Walldrywall4Tiled"}
                exterior_wall["polygon"] = wall["polygon"][::-1]
                exterior_wall["segment"] = wall["segment"][::-1]
                wall["connect_exterior"] = exterior_wall["id"]
                updated_walls.append(exterior_wall)
            updated_walls.append(wall)
        walls = updated_walls

        return wall_height, walls

    def get_wall_height(self, scene):
        # get wall height
        wall_height_prompt = self.wall_height_template.format(input=scene["query"])

        if "wall_height" not in scene:
            wall_height = self.llm(wall_height_prompt).split("\n")[0].strip()

            try:
                wall_height = float(wall_height)
            except:
                wall_height = round(
                    random.uniform(2.5, 4.5), 1
                )  # if failed, random height between 2.5 and 4.5

            scene["wall_height"] = min(
                max(wall_height, 2.0), 4.5
            )  # limit the wall height between 2.0 and 4.5

        wall_height = scene["wall_height"]
        print(f"\nUser: {wall_height_prompt}\n")
        print(f"{Fore.GREEN}AI: The wall height is {wall_height}{Fore.RESET}")

        return wall_height

    def generate_wall_polygon(self, point, next_point, wall_height):
        wall_polygon = []
        # add the base point
        wall_polygon.append({"x": point[0], "y": 0, "z": point[1]})
        # add the top point (with the same x and z, but y = wall_height)
        wall_polygon.append({"x": point[0], "y": wall_height, "z": point[1]})
        # add the top point of the next base point
        wall_polygon.append({"x": next_point[0], "y": wall_height, "z": next_point[1]})
        # add the next base point
        wall_polygon.append({"x": next_point[0], "y": 0, "z": next_point[1]})
        return wall_polygon

    def get_connected_rooms(self, wall_polygon, rooms, roomId):
        connected_rooms = []
        vertices0 = [
            (vertex["x"], vertex["z"]) for vertex in wall_polygon if vertex["y"] == 0
        ]
        lines0 = [LineString([vertices0[0], vertices0[1]])]

        for room in rooms:
            if room["id"] == roomId:
                continue  # do not consider the room itself

            room_polygon = room["floorPolygon"]
            vertices1 = [(vertex["x"], vertex["z"]) for vertex in room_polygon]
            lines1 = [
                LineString([vertices1[i], vertices1[(i + 1) % len(vertices1)]])
                for i in range(len(vertices1))
            ]

            shared_segments = self.check_connected(lines0, lines1)

            if shared_segments != None:
                connected_room = shared_segments[0]
                connected_room["roomId"] = room["id"]
                connected_rooms.append(connected_room)

        return connected_rooms

    def check_connected(self, lines0, lines1):
        shared_segments = []
        for line0 in lines0:
            for line1 in lines1:
                if line0.intersects(line1):
                    intersection = line0.intersection(line1)
                    if intersection.geom_type == "LineString":
                        shared_segments.append(
                            {
                                "intersection": [
                                    {
                                        "x": intersection.xy[0][0],
                                        "y": 0,
                                        "z": intersection.xy[1][0],
                                    },
                                    {
                                        "x": intersection.xy[0][1],
                                        "y": 0,
                                        "z": intersection.xy[1][1],
                                    },
                                ],
                                "line0": [
                                    {"x": line0.xy[0][0], "y": 0, "z": line0.xy[1][0]},
                                    {"x": line0.xy[0][1], "y": 0, "z": line0.xy[1][1]},
                                ],
                                "line1": [
                                    {"x": line1.xy[0][0], "y": 0, "z": line1.xy[1][0]},
                                    {"x": line1.xy[0][1], "y": 0, "z": line1.xy[1][1]},
                                ],
                            }
                        )

        # Return shared line segments, if any
        if shared_segments:
            return shared_segments

        # If no shared line segments, return None
        return None

    def update_walls(self, original_walls, open_room_pairs):
        # update walls since there could be open connections
        updated_walls = []
        deleted_wallIds = []
        for wall in original_walls:
            room0_id = wall["roomId"]
            connection = wall["connected_rooms"]
            if len(connection) == 0:
                updated_walls.append(wall)
            else:
                room1_id = connection[0]["roomId"]
                if (room0_id, room1_id) in open_room_pairs or (
                    room1_id,
                    room0_id,
                ) in open_room_pairs:
                    deleted_wallIds.append(wall["id"])
                else:
                    updated_walls.append(wall)

        # create bounding box for open connections
        open_wall_segments = []
        for wallId in deleted_wallIds:
            wall = [wall for wall in original_walls if wall["id"] == wallId][0]
            open_wall_segments.append(wall["segment"])

        open_wall_segments_no_overlap = []
        for segment in open_wall_segments:
            if (
                segment not in open_wall_segments_no_overlap
                and segment[::-1] not in open_wall_segments_no_overlap
            ):
                open_wall_segments_no_overlap.append(segment)

        open_wall_rectangles = []
        for segment in open_wall_segments_no_overlap:
            top_rectangle, bottom_rectangle = self.create_rectangles(segment)
            open_wall_rectangles.append(top_rectangle)
            open_wall_rectangles.append(bottom_rectangle)

        open_walls = {
            "segments": open_wall_segments_no_overlap,
            "openWallBoxes": open_wall_rectangles,
        }

        return updated_walls, open_walls

    def get_wall_direction(self, wall_endpoint1, wall_endpoint2, room_vertices):
        wall_width = np.linalg.norm(np.array(wall_endpoint1) - np.array(wall_endpoint2))

        wall_direction = None
        room_polygon = Polygon(room_vertices)
        wall_center = [
            (wall_endpoint1[0] + wall_endpoint2[0]) / 2,
            (wall_endpoint1[1] + wall_endpoint2[1]) / 2,
        ]

        if wall_endpoint1[1] == wall_endpoint2[1]:
            extend_point_1 = [wall_center[0], wall_center[1] + 0.01]
            extend_point_2 = [wall_center[0], wall_center[1] - 0.01]
            # check which point is in room polygon
            if room_polygon.contains(Point(extend_point_1)):
                wall_direction = "south"
            elif room_polygon.contains(Point(extend_point_2)):
                wall_direction = "north"

        elif wall_endpoint1[0] == wall_endpoint2[0]:
            extend_point_1 = [wall_center[0] + 0.01, wall_center[1]]
            extend_point_2 = [wall_center[0] - 0.01, wall_center[1]]
            # check which point is in room polygon
            if room_polygon.contains(Point(extend_point_1)):
                wall_direction = "west"
            elif room_polygon.contains(Point(extend_point_2)):
                wall_direction = "east"

        return wall_width, wall_direction

    def create_rectangles(self, segment):
        # Convert to numpy arrays for easier calculations
        pt1 = np.array(segment[0])
        pt2 = np.array(segment[1])

        # Calculate the vector for the segment
        vec = pt2 - pt1

        # Calculate a perpendicular vector with length 1
        perp_vec = np.array([-vec[1], vec[0]], dtype=np.float32)
        perp_vec /= np.linalg.norm(perp_vec)
        perp_vec *= (
            0.5  # 0.5 is the hyperparameter for the width of the open connection
        )

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
