import ast
import copy
import math
import json
import torch
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
from colorama import Fore
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import modules.prompts as prompts
import matplotlib.colors as mcolors
from difflib import SequenceMatcher
from langchain import PromptTemplate
from shapely.geometry import LineString, Point, Polygon


class FloorPlanGenerator():
    def __init__(self, clip_model, clip_process, clip_tokenizer, llm):
        self.json_template = {"ceilings": [], "children": [], "vertices": None,
                              "floorMaterial": {"name": None, "color": None}, 
                              "floorPolygon": [], "id": None, "roomType": None}
        self.material_selector = MaterialSelector(clip_model, clip_process, clip_tokenizer)
        self.floor_plan_template = PromptTemplate(input_variables=["input", "additional_requirements"], template=prompts.floor_plan_prompt)
        self.llm = llm
        self.used_assets = []

    
    def generate_rooms(self, scene, additional_requirements="N/A", visualize=False):
        # get floor plan if not provided
        floor_plan_prompt = self.floor_plan_template.format(input=scene["query"], additional_requirements=additional_requirements)
        if "raw_floor_plan" not in scene:
            raw_floor_plan = self.llm(floor_plan_prompt)
            scene["raw_floor_plan"] = raw_floor_plan
        else:
            raw_floor_plan = scene["raw_floor_plan"]
        
        print(f"User: {floor_plan_prompt}\n")
        print(f"{Fore.GREEN}AI: Here is the floor plan:\n{raw_floor_plan}{Fore.RESET}")
        
        rooms = self.get_plan(scene["query"], scene["raw_floor_plan"], visualize)
        return rooms


    def get_plan(self, query, raw_plan, visualize=False):
        parsed_plan = self.parse_raw_plan(raw_plan)

        # select materials
        all_designs = []
        for room in parsed_plan:
            all_designs.append(room["floor_design"])
            all_designs.append(room["wall_design"])
        design2material = self.select_materials(all_designs, topk=5)
        
        # assign materials
        for i in range(len(parsed_plan)):
            parsed_plan[i]["floorMaterial"] = design2material[parsed_plan[i]["floor_design"]]
            parsed_plan[i]["wallMaterial"] = design2material[parsed_plan[i]["wall_design"]]

        if visualize: self.visualize_floor_plan(query, parsed_plan)

        return parsed_plan
        

    def parse_raw_plan(self, raw_plan):
        parsed_plan = []
        room_types = []
        plans = [plan.lower() for plan in raw_plan.split("\n") if "|" in plan]
        for i, plan in enumerate(plans):
            room_type, floor_design, wall_design, vertices = plan.split("|")
            room_type = room_type.strip().replace("'", "") # remove single quote

            if room_type in room_types: room_type += f"-{i}"
            room_types.append(room_type)

            floor_design = floor_design.strip()
            wall_design = wall_design.strip()
            vertices = ast.literal_eval(vertices.strip())
            # change to float
            vertices = [(float(vertex[0]), float(vertex[1])) for vertex in vertices]

            current_plan = copy.deepcopy(self.json_template)
            current_plan["id"] = room_type
            current_plan["roomType"] = room_type
            current_plan["vertices"], current_plan["floorPolygon"] = self.vertices2xyz(vertices)
            current_plan["floor_design"] = floor_design
            current_plan["wall_design"] = wall_design
            parsed_plan.append(current_plan)

        # get full vertices: consider the intersection with other rooms
        all_vertices = []
        for room in parsed_plan:
            all_vertices += room["vertices"]
        all_vertices = list(set(map(tuple, all_vertices)))

        for room in parsed_plan:
            full_vertices = self.get_full_vertices(room["vertices"], all_vertices)
            full_vertices = list(set(map(tuple, full_vertices)))
            room["full_vertices"], room["floorPolygon"] = self.vertices2xyz(full_vertices)
        
        valid, msg = self.check_validity(parsed_plan)

        if not valid: print(f"{Fore.RED}AI: {msg}{Fore.RESET}"); raise ValueError(msg)
        else: print(f"{Fore.GREEN}AI: {msg}{Fore.RESET}"); return parsed_plan
    

    def vertices2xyz(self, vertices):
        sort_vertices = self.sort_vertices(vertices)
        xyz_vertices = [{"x": vertex[0], "y": 0, "z": vertex[1]} for vertex in sort_vertices]
        return sort_vertices, xyz_vertices
    

    def xyz2vertices(self, xyz_vertices):
        vertices = [(vertex["x"], vertex["z"]) for vertex in xyz_vertices]
        return vertices


    def sort_vertices(self, vertices):
        # Calculate the centroid of the polygon
        cx = sum(x for x, y in vertices) / max(len(vertices), 1)
        cy = sum(y for x, y in vertices) / max(len(vertices), 1)

        # Sort the vertices in clockwise order
        vertices_clockwise = sorted(vertices, key=lambda v: (-math.atan2(v[1]-cy, v[0]-cx)) % (2*math.pi))

        # Find the vertex with the smallest x value
        min_vertex = min(vertices_clockwise, key=lambda v: v[0])

        # Rotate the vertices so the vertex with the smallest x value is first
        min_index = vertices_clockwise.index(min_vertex)
        vertices_clockwise = vertices_clockwise[min_index:] + vertices_clockwise[:min_index]

        return vertices_clockwise


    def get_full_vertices(self, original_vertices, all_vertices):
        # Create line segments from the original vertices
        lines = [LineString([original_vertices[i], original_vertices[(i+1)%len(original_vertices)]]) for i in range(len(original_vertices))]

        # Check each vertex against each line segment
        full_vertices = []
        for vertex in all_vertices:
            point = Point(vertex)
            for line in lines:
                if line.intersects(point):
                    full_vertices.append(vertex)

        return full_vertices


    def select_materials(self, designs, topk):
        candidate_materials = self.material_selector.match_material(designs, topk=topk)[0]
        candidate_colors = self.material_selector.select_color(designs, topk=topk)[0]
        
        # remove used materials
        top_materials = [[materials[0]] for materials in candidate_materials]
        candidate_materials = [[material for material in materials if material not in self.used_assets] for materials in candidate_materials]

        selected_materials = []
        for i in range(len(designs)):
            if len(candidate_materials[i]) == 0: selected_materials.append(top_materials[i][0])
            else: selected_materials.append(candidate_materials[i][0])

        selected_colors = [candidates[0] for candidates in candidate_colors]

        design2materials = {design: {} for design in designs}
        for i, design in enumerate(designs):
            design2materials[design]["name"] = selected_materials[i]
            # design2materials[design]["color"] = self.color2rgb(selected_colors[i])
        return design2materials
    

    def color2rgb(self, color_name):
        rgb = mcolors.to_rgb(color_name)
        return {"r": rgb[0], "g": rgb[1], "b": rgb[2]}


    def parsed2raw(self, rooms):
        raw_plan = ""
        for room in rooms:
            raw_plan += " | ".join([room["roomType"], room["floor_design"], room["wall_design"], str(room["vertices"])])
            raw_plan += "\n"
        return raw_plan
    

    def check_interior_angles(self, vertices):
        n = len(vertices)
        for i in range(n):
            a, b, c = vertices[i], vertices[(i + 1) % n], vertices[(i + 2) % n]
            angle = abs(math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])))
            if angle < 90 or angle > 270:
                return False
        return True


    def check_validity(self, rooms):
        room_polygons = [Polygon(room["vertices"]) for room in rooms]

        # check interior angles
        for room in rooms:
            if not self.check_interior_angles(room["vertices"]):
                return False, "All interior angles of the room must be greater than or equal to 90 degrees."
                
        if len(room_polygons) == 1: 
            return True, "The floor plan is valid. (Only one room)"
                
        # check overlap, connectivity and vertex inside another room
        for i in range(len(room_polygons)):
            has_neighbor = False
            for j in range(len(room_polygons)):
                if i != j:
                    if room_polygons[i].equals(room_polygons[j]) or room_polygons[i].contains(room_polygons[j]) or room_polygons[j].contains(room_polygons[i]):
                        return False, "Room polygons must not overlap."
                    intersection = room_polygons[i].intersection(room_polygons[j])
                    if isinstance(intersection, LineString):
                        has_neighbor = True
                    for vertex in rooms[j]["vertices"]:
                        if Polygon(rooms[i]["vertices"]).contains(Point(vertex)):
                            return False, "No vertex of a room can be inside another room."
            if not has_neighbor:
                return False, "Each room polygon must share an edge with at least one other room polygon."

        return True, "The floor plan is valid."



    def visualize_floor_plan(self, query, parsed_plan):
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 22
        fig, ax = plt.subplots(figsize=(10, 10))
        colors = [
            (0.53, 0.81, 0.98, 0.5),
            (0.56, 0.93, 0.56, 0.5),
            (0.94, 0.5, 0.5, 0.5),
            (1.0, 1.0, 0.88, 0.5),
        ]

        def midpoint(p1, p2):
            return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

        for i, room in enumerate(parsed_plan):
            coordinates = room["vertices"]
            polygon = patches.Polygon(coordinates, closed=True, edgecolor='black', linewidth=2)
            polygon.set_facecolor(colors[i % len(colors)])
            ax.add_patch(polygon)

        for i, room in enumerate(parsed_plan):
            coordinates = room["vertices"]
            # Label the rooms
            x, y = zip(*coordinates)
            room_x = sum(x) / len(coordinates)
            room_y = sum(y) / len(coordinates)
            # ax.text(room_x, room_y, room["roomType"], ha='center', va='center')

            # Add points to the corners
            ax.scatter(x, y, s=100, color='black')  # s is the size of the point

            # # Display width and length
            # for i in range(len(coordinates)):
            #     p1, p2 = coordinates[i], coordinates[(i + 1) % len(coordinates)]
            #     label = f"{np.round(np.linalg.norm(np.array(p1) - np.array(p2)), 2)} m"
            #     ax.text(*midpoint(p1, p2), label, ha='center', va='center', fontsize=20, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round4'))

        # Set aspect of the plot to be equal, so squares appear as squares
        ax.set_aspect('equal')
        ax.autoscale_view()

        # Turn off the axis
        ax.axis('off')

        folder_name = query.replace(" ", "_")
        plt.savefig(f"{folder_name}.pdf", bbox_inches='tight', dpi=300)
        plt.show()
    

class MaterialSelector():
    def __init__(self, clip_model, clip_preprocess, clip_tokenizer):
        materials = json.load(open("data/materials/material-database.json", "r"))
        self.selected_materials = materials["Wall"] + materials["Wood"] + materials["Fabric"]
        self.colors = list(mcolors.CSS4_COLORS.keys())

        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.clip_tokenizer = clip_tokenizer

        self.load_features()


    def load_features(self):        
        try:
            self.material_feature_clip = pickle.load(open("data/materials/material_feature_clip.p", "rb"))
        except:
            print("Precompute image features for materials...")
            self.material_feature_clip = []
            for material in tqdm(self.selected_materials):
                image = self.preprocess(Image.open(f"data/materials/images/{material}.png")).unsqueeze(0)
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(image)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                self.material_feature_clip.append(image_features)
            self.material_feature_clip = torch.vstack(self.material_feature_clip)
            pickle.dump(self.material_feature_clip, open("data/materials/material_feature_clip.p", "wb"))
        
        try:
            self.color_feature_clip = pickle.load(open("data/materials/color_feature_clip.p", "rb"))
        except:
            print("Precompute text features for colors...")
            with torch.no_grad():
                self.color_feature_clip = self.clip_model.encode_text(self.clip_tokenizer(self.colors))
                self.color_feature_clip /= self.color_feature_clip.norm(dim=-1, keepdim=True)
            pickle.dump(self.color_feature_clip, open("data/materials/color_feature_clip.p", "wb"))


    def match_material(self, queries, topk=5):
        with torch.no_grad():
            query_feature_clip = self.clip_model.encode_text(self.clip_tokenizer(queries))
            query_feature_clip /= query_feature_clip.norm(dim=-1, keepdim=True)
        
        clip_similarity = query_feature_clip @ self.material_feature_clip.T
        string_similarity = torch.tensor([[self.string_match(query, material) for material in self.selected_materials] for query in queries])
        
        joint_similarity = string_similarity + clip_similarity # use visual embedding only seems to be better

        results = []
        scores = []
        for sim in joint_similarity:
            indices = torch.argsort(sim, descending=True)[:topk]
            results.append([self.selected_materials[ind] for ind in indices])
            scores.append([sim[ind] for ind in indices])
        return results, scores


    def select_color(self, queries, topk=5):
        with torch.no_grad():
            query_feature_clip = self.clip_model.encode_text(self.clip_tokenizer(queries))
            query_feature_clip /= query_feature_clip.norm(dim=-1, keepdim=True)
        clip_similarity = query_feature_clip @ self.color_feature_clip.T

        results = []
        scores = []
        for sim in clip_similarity:
            indices = torch.argsort(sim, descending=True)[:topk]
            results.append([self.colors[ind] for ind in indices])
            scores.append([sim[ind] for ind in indices])
        return results, scores
    

    def string_match(self, a, b):
        return SequenceMatcher(None, a, b).ratio()