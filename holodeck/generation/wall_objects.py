import copy
import multiprocessing
import random
import re
import time

import matplotlib.pyplot as plt
import numpy as np
from langchain import PromptTemplate, OpenAI
from shapely.geometry import Polygon, box, Point, LineString
from shapely.ops import substring

import holodeck.generation.prompts as prompts
from holodeck.generation.objaverse_retriever import ObjathorRetriever
from holodeck.generation.utils import get_bbox_dims


class WallObjectGenerator():
    def __init__(self, object_retriever: ObjathorRetriever, llm: OpenAI):
        self.json_template = {"assetId": None, "id": None, "kinematic": True,
                              "position": {}, "rotation": {}, "material": None, "roomId": None}
        self.llm = llm
        self.object_retriever = object_retriever
        self.database = object_retriever.database
        self.constraint_prompt_template = PromptTemplate(input_variables=["room_type", "wall_height", "floor_objects", "wall_objects"],
                                                template=prompts.wall_object_constraints_prompt)
        self.grid_size = 25
        self.default_height = 150
        self.constraint_type = "llm"


    def generate_wall_objects(self, scene, use_constraint=True):
        doors = scene["doors"]
        windows = scene["windows"]
        open_walls = scene["open_walls"]
        wall_height = scene["wall_height"]

        wall_objects = []
        selected_objects = scene["selected_objects"]

        packed_args = [(room, scene, doors, windows, open_walls, wall_height, selected_objects, use_constraint) for room in scene["rooms"]]
        pool = multiprocessing.Pool(processes=4)
        all_placements = pool.map(self.generate_wall_objects_per_room, packed_args)
        pool.close()
        pool.join()

        for placements in all_placements:
            wall_objects += placements

        return wall_objects
    

    def generate_wall_objects_per_room(self, args):
        room, scene, doors, windows, open_walls, wall_height, selected_objects, use_constraint = args

        selected_wall_objects = selected_objects[room["roomType"]]["wall"]
        selected_wall_objects = self.order_objects_by_size(selected_wall_objects)
        wall_object_name2id = {object_name: asset_id for object_name, asset_id in selected_wall_objects}

        room_id = room["id"]
        room_type = room["roomType"]

        wall_object_names = list(wall_object_name2id.keys())
        
        floor_object_name2id = {object["object_name"]: object["assetId"] for object in scene["floor_objects"] if object["roomId"] == room["id"]}
        floor_object_names = list(floor_object_name2id.keys())
        
        # get constraints
        constraints_prompt = self.constraint_prompt_template.format(room_type=room_type,
                                                                    wall_height=int(wall_height*100),
                                                                    floor_objects=", ".join(floor_object_names),
                                                                    wall_objects=", ".join(wall_object_names))
        if self.constraint_type == "llm" and use_constraint:
            constraint_plan = self.llm(constraints_prompt)
        else:
            constraint_plan = ""
            for object_name in wall_object_names:
                random_height = random.randint(0, int(wall_height*100))
                constraint_plan += f"{object_name} | N/A | {random_height} \n"

        print(f"\nwall object constraint plan for {room_type}:\n{constraint_plan}")
        constraints = self.parse_wall_object_constraints(constraint_plan, wall_object_names, floor_object_names)
        
        # get wall objects
        wall_object2dimension = {object_name: get_bbox_dims(self.database[object_id]) for object_name, object_id in wall_object_name2id.items()}
        wall_objects_list = [(object_name, (wall_object2dimension[object_name]['x'] * 100, wall_object2dimension[object_name]['y'] * 100, wall_object2dimension[object_name]['z'] * 100)) for object_name in constraints]
        
        # update constraints with max height
        wall_object2max_height = {object_name: min(scene["wall_height"] * 100 - wall_object2dimension[object_name]["y"] * 100 - 20, constraints[object_name]["height"]) for object_name in constraints}
        for object_name in constraints:
            constraints[object_name]["height"] = max(wall_object2max_height[object_name], 0) # avoid negative height
        
        # get initial state
        room_vertices = [(x * 100, y * 100) for (x, y) in room["vertices"]]
        room_poly = Polygon(room_vertices)
        initial_state = self.get_initial_state(scene, doors, windows, room_vertices, open_walls)

        # solve
        room_x, room_z = self.get_room_size(room)
        grid_size = max(room_x // 20, room_z // 20)

        solver = DFS_Solver_Wall(grid_size=grid_size, max_duration=5, constraint_bouns=100)
        solutions = solver.get_solution(room_poly, wall_objects_list, constraints, initial_state)
        
        placements = self.solution2placement(solutions, wall_object_name2id, room_id)

        return placements


    def parse_wall_object_constraints(self, constraint_text, wall_object_names, floor_object_names):
        object2constraints = {}
        lines = [line.lower() for line in constraint_text.split('\n') if "|" in line]
        for line in lines:
            # remove index
            pattern = re.compile(r'^\d+\.\s*')
            line = pattern.sub('', line)
            if line[-1] == ".": line = line[:-1] # remove the last period
            try:
                object_name, location, height = line.split("|")
                object_name = object_name.replace("*", "").strip()
                location = location.strip()
                height = height.strip()
            except:
                print(f"Warning: cannot parse {line}.")
                continue
            
            if object_name not in wall_object_names: continue

            try: target_floor_object_name = location.split(", ")[-1]
            except: print(f"Warning: cannot parse {location}."); target_floor_object_name = None

            try: height = int(height)
            except: height = self.default_height
            
            if target_floor_object_name in floor_object_names:
                object2constraints[object_name] = {"target_floor_object_name": target_floor_object_name, "height": height}
            else:
                object2constraints[object_name] = {"target_floor_object_name": None, "height": height}
        
        return object2constraints
    

    def get_room_size(self, room):
        floor_polygon = room["floorPolygon"]
        x_values = [point['x'] for point in floor_polygon]
        z_values = [point['z'] for point in floor_polygon]
        return (int(max(x_values) - min(x_values)) * 100, int(max(z_values) - min(z_values)) * 100)


    def check_wall_object_size(self, room_size, object_size):
        if object_size["x"] * 100 > max(room_size) * 0.5:
            print(f"Warning: object size {object_size} is too large for room size {room_size}.")
            return False
        else:
            return True
    

    def get_initial_state(self, scene, doors, windows, room_vertices, open_walls):
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
                    initial_state[f"door-{i}"] = ((x_min, 0, z_min), (x_max, door_height, z_max), 0, door_vertices, 1)
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
                    initial_state[f"window-{i}"] = ((x_min, y_min, z_min), (x_max, y_max, z_max), 0, window_vertices, 1)
                    i += 1
        
        if len(open_walls) != 0:
            open_wall_boxes = open_walls["openWallBoxes"]
            for open_wall_box in open_wall_boxes:
                open_wall_vertices = [(x * 100, z * 100) for (x, z) in open_wall_box]
                open_wall_poly = Polygon(open_wall_vertices)
                open_wall_center = open_wall_poly.centroid
                if room_poly.contains(open_wall_center):
                    x_min, z_min, x_max, z_max = open_wall_poly.bounds
                    initial_state[f"open-{i}"] = ((x_min, 0, z_min), (x_max, scene["wall_height"] * 100, z_max), 0, open_wall_vertices, 1)
                    i += 1
        
        for object in scene["floor_objects"]:
            try: object_vertices = object["vertices"]
            except: continue
            
            object_poly = Polygon(object_vertices)
            object_center = object_poly.centroid
            if room_poly.contains(object_center):
                object_height = object["position"]["y"] * 100 * 2 # the height should be twice the value of the y coordinate
                x_min, z_min, x_max, z_max = object_poly.bounds
                initial_state[object["object_name"]] = ((x_min, 0, z_min), (x_max, object_height, z_max), object["rotation"]["y"], object_vertices, 1)

        return initial_state
    

    def solution2placement(self, solutions, wall_object_name2id, room_id):
        placements = []
        for object_name, solution in solutions.items():
            if object_name not in wall_object_name2id: continue
            placement = self.json_template.copy()
            placement["assetId"] = wall_object_name2id[object_name]
            placement["id"] = f"{object_name} ({room_id})"
            position_x = (solution[0][0] + solution[1][0]) / 200
            position_y = (solution[0][1] + solution[1][1]) / 200
            position_z = (solution[0][2] + solution[1][2]) / 200
            
            placement["position"] = {"x": position_x, "y": position_y, "z": position_z}
            placement["rotation"] = {"x": 0, "y": solution[2], "z": 0}
            
            # move the object a little bit to avoid collision
            if placement["rotation"]["y"] == 0: placement["position"]["z"] += 0.01
            elif placement["rotation"]["y"] == 90: placement["position"]["x"] += 0.01
            elif placement["rotation"]["y"]== 180: placement["position"]["z"] -= 0.01
            elif placement["rotation"]["y"] == 270: placement["position"]["x"] -= 0.01

            placement["roomId"] = room_id
            placement["vertices"] = list(solution[3])
            placement["object_name"] = object_name
            placements.append(placement)
        return placements
    

    def order_objects_by_size(self, selected_wall_objects):
        ordered_wall_objects = []
        for object_name, asset_id in selected_wall_objects:
            dimensions = get_bbox_dims(self.database[asset_id])
            size = dimensions["x"]
            ordered_wall_objects.append([object_name, asset_id, size])
        ordered_wall_objects.sort(key=lambda x: x[2], reverse=True)
        ordered_wall_objects_no_size = [[object_name, asset_id] for object_name, asset_id, size in ordered_wall_objects]
        return ordered_wall_objects_no_size


class SolutionFound(Exception):
    def __init__(self, solution):
        self.solution = solution
        pass


class DFS_Solver_Wall():
    def __init__(self, grid_size, random_seed=0, max_duration=5, constraint_bouns=100):
        self.grid_size = grid_size
        self.random_seed = random_seed
        self.max_duration = max_duration  # maximum allowed time in seconds
        self.constraint_bouns = constraint_bouns
        self.start_time = None
        self.solutions = []
        self.visualize = False


    def get_solution(self, room_poly, wall_objects_list, constraints, initial_state):
        grid_points = self.create_grids(room_poly)

        self.start_time = time.time()
        try:
            self.dfs(room_poly, wall_objects_list, constraints, grid_points, initial_state)
        except SolutionFound as e:
            print(f"Time taken: {time.time() - self.start_time}")
        
        max_solution = self.get_max_solution(self.solutions)
        
        if self.visualize: self.visualize_grid(room_poly, grid_points, max_solution)
        return max_solution


    def get_max_solution(self, solutions):
        path_weights = []
        for solution in solutions:
            path_weights.append(sum([obj[-1] for obj in solution.values()]))
        max_index = np.argmax(path_weights)
        return solutions[max_index]


    def dfs(self, room_poly, wall_objects_list, constraints, grid_points, placed_objects):
        if len(wall_objects_list) == 0:
            self.solutions.append(placed_objects)
            return placed_objects
        
        if time.time() - self.start_time > self.max_duration:
            print(f"Time limit reached.")
            raise SolutionFound(self.solutions)
        
        object_name, object_dim = wall_objects_list[0]
        placements = self.get_possible_placements(room_poly, object_dim, constraints[object_name], grid_points, placed_objects)
        
        if len(placements) == 0:
            self.solutions.append(placed_objects)

        paths = []
        for placement in placements:
            placed_objects_updated = copy.deepcopy(placed_objects)
            placed_objects_updated[object_name] = placement

            sub_paths = self.dfs(room_poly, wall_objects_list[1:], constraints, grid_points, placed_objects_updated)
            paths.extend(sub_paths)

        return paths
    

    def get_possible_placements(self, room_poly, object_dim, constraint, grid_points, placed_objects):
        all_solutions = self.filter_collision(placed_objects, self.get_all_solutions(room_poly, grid_points, object_dim, constraint["height"]))
        random.shuffle(all_solutions)
        target_floor_object_name = constraint["target_floor_object_name"]
        if target_floor_object_name is not None and target_floor_object_name in placed_objects:
            all_solutions = self.score_solution_by_distance(all_solutions, placed_objects[target_floor_object_name])
            # order solutions by distance to target floor object
            all_solutions = sorted(all_solutions, key=lambda x: x[-1], reverse=True)
        return all_solutions


    def create_grids(self, room_poly):
        # Get the coordinates of the polygon
        poly_coords = list(room_poly.exterior.coords)

        grid_points = []
        # Iterate over each pair of points (edges of the polygon)
        for i in range(len(poly_coords) - 1):
            line = LineString([poly_coords[i], poly_coords[i + 1]])
            line_length = line.length

            # Create points along the edge at intervals of grid size
            for j in range(0, int(line_length), self.grid_size):
                point_on_line = substring(line, j, j) # Get a point at distance j from the start of the line
                if point_on_line:
                    grid_points.append((point_on_line.x, point_on_line.y))
        
        return grid_points
    

    def get_all_solutions(self, room_poly, grid_points, object_dim, height):
        obj_length, obj_height, obj_width = object_dim
        obj_half_length = obj_length / 2

        rotation_adjustments = {
            0: ((-obj_half_length, 0), (obj_half_length, obj_width)),
            90: ((0, -obj_half_length), (obj_width, obj_half_length)),
            180: ((-obj_half_length, -obj_width), (obj_half_length, 0)),
            270: ((-obj_width, -obj_half_length), (0, obj_half_length))
        }

        solutions = []
        for rotation in [0, 90, 180, 270]:
            for point in grid_points:
                center_x, center_y = point
                lower_left_adjustment, upper_right_adjustment = rotation_adjustments[rotation]
                lower_left = (center_x + lower_left_adjustment[0], center_y + lower_left_adjustment[1])
                upper_right = (center_x + upper_right_adjustment[0], center_y + upper_right_adjustment[1])
                obj_box = box(*lower_left, *upper_right)

                if room_poly.contains(obj_box):
                    object_coords = obj_box.exterior.coords[:]
                    coordinates_on_edge = [coord for coord in object_coords if room_poly.boundary.contains(Point(coord))]
                    coordinates_on_edge = list(set(coordinates_on_edge))
                    if len(coordinates_on_edge) >= 2:
                        vertex_min = (lower_left[0], height, lower_left[1])
                        vertex_max = (upper_right[0], height + obj_height, upper_right[1])

                        solutions.append([vertex_min, vertex_max, rotation, tuple(obj_box.exterior.coords[:]), 1])
                    
        return solutions
    

    def filter_collision(self, placed_objects, solutions):
        def intersect_3d(box1, box2):
            # box1 and box2 are dictionaries with 'min' and 'max' keys,
            # which are tuples representing the minimum and maximum corners of the 3D box.
            for i in range(3):
                if box1['max'][i] < box2['min'][i] or box1['min'][i] > box2['max'][i]:
                    return False
            return True

        valid_solutions = []
        boxes = [{"min": vertex_min, "max": vertex_max} for vertex_min, vertex_max, rotation, box_coords, path_weight in placed_objects.values()]

        for solution in solutions:
            for box in boxes:
                if intersect_3d(box, {"min": solution[0], "max": solution[1]}):
                    break
            else:
                valid_solutions.append(solution)
        
        return valid_solutions
    

    def score_solution_by_distance(self, solutions, target_object):
        distances = []
        scored_solutions = []
        for solution in solutions:
            center_x, center_y, center_z = (solution[0][0]+solution[1][0])/2, (solution[0][1]+solution[1][1])/2, (solution[0][2]+solution[1][2])/2
            target_x, target_y, target_z = (target_object[0][0]+target_object[1][0])/2, (target_object[0][1]+target_object[1][1])/2, (target_object[0][2]+target_object[1][2])/2
            distance = np.sqrt((center_x - target_x)**2 + (center_y - target_y)**2 + (center_z - target_z)**2)
            distances.append(distance)
            scored_solution = solution.copy()
            scored_solution[-1] = solution[-1] + self.constraint_bouns * (1/distance)
            scored_solutions.append(scored_solution)
        return scored_solutions
        
    
    def visualize_grid(self, room_poly, grid_points, solutions):
        # create a new figure
        fig, ax = plt.subplots()

        # draw the room
        x, y = room_poly.exterior.xy
        ax.plot(x, y, 'b-', label='Room')

        # draw the grid points
        grid_x = [point[0] for point in grid_points]
        grid_y = [point[1] for point in grid_points]
        ax.plot(grid_x, grid_y, 'ro', markersize=2)

        # draw the solutions
        for object_name, solution in solutions.items():
            vertex_min, vertex_max, rotation, box_coords = solution[:-1]
            center_x, center_y = (vertex_min[0]+vertex_max[0])/2, (vertex_min[2]+vertex_max[2])/2

            # create a polygon for the solution
            obj_poly = Polygon(box_coords)
            x, y = obj_poly.exterior.xy
            ax.plot(x, y, 'g-', linewidth=2)

            ax.text(center_x, center_y, object_name, fontsize=12, ha='center')

            # set arrow direction based on rotation
            if rotation == 0:
                ax.arrow(center_x, center_y, 0, 25, head_width=10, fc='g')
            elif rotation == 90:
                ax.arrow(center_x, center_y, 25, 0, head_width=10, fc='g')
            elif rotation == 180:
                ax.arrow(center_x, center_y, 0, -25, head_width=10, fc='g')
            elif rotation == 270:
                ax.arrow(center_x, center_y, -25, 0, head_width=10, fc='g')

        ax.set_aspect('equal', 'box')  # to keep the ratios equal along x and y axis
        plt.show()