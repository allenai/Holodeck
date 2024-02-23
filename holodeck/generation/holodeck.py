import datetime
import os
from typing import Optional, Dict, Any, Tuple

import compress_json
import open_clip
from langchain.llms import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from holodeck.constants import (
    HOLODECK_BASE_DATA_DIR,
    OBJATHOR_VERSIONED_DIR,
    OBJATHOR_ASSETS_DIR,
    OBJATHOR_FEATURES_DIR,
    OBJATHOR_ANNOTATIONS_PATH,
    HOLODECK_THOR_FEATURES_DIR,
    HOLODECK_THOR_ANNOTATIONS_PATH,
)
from holodeck.generation.ceiling_objects import CeilingObjectGenerator
from holodeck.generation.doors import DoorGenerator
from holodeck.generation.floor_objects import FloorObjectGenerator
from holodeck.generation.layers import map_asset2layer
from holodeck.generation.lights import generate_lights
from holodeck.generation.objaverse_retriever import ObjathorRetriever
from holodeck.generation.object_selector import ObjectSelector
from holodeck.generation.rooms import FloorPlanGenerator
from holodeck.generation.skybox import getSkybox
from holodeck.generation.small_objects import SmallObjectGenerator
from holodeck.generation.utils import get_top_down_frame, room_video
from holodeck.generation.wall_objects import WallObjectGenerator
from holodeck.generation.walls import WallGenerator
from holodeck.generation.windows import WindowGenerator


def confirm_paths_exist():
    for p in [
        OBJATHOR_VERSIONED_DIR,
        OBJATHOR_ASSETS_DIR,
        OBJATHOR_FEATURES_DIR,
        OBJATHOR_ANNOTATIONS_PATH,
        HOLODECK_BASE_DATA_DIR,
        HOLODECK_THOR_FEATURES_DIR,
        HOLODECK_THOR_ANNOTATIONS_PATH,
    ]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Path {p} does not exist, this must exist for Holodeck generation to succeed."
                f" Please see the Holodeck README file at https://github.com/allenai/Holodeck/blob/main/README.md"
                f" for instruction on how to set up the required data directories."
            )


class Holodeck:
    def __init__(
        self,
        openai_api_key: str,
        openai_org: Optional[str],
        objaverse_asset_dir: str,
        single_room,
    ):
        confirm_paths_exist()

        if openai_org is not None:
            os.environ["OPENAI_ORG"] = openai_org

        # initialize llm
        self.llm = OpenAI(
            model_name="gpt-4-1106-preview",
            max_tokens=2048,
            openai_api_key=openai_api_key,
        )
        self.llm_fast = OpenAI(
            model_name="gpt-3.5-turbo",
            max_tokens=2048,
            openai_api_key=openai_api_key,
        )

        # initialize CLIP
        (
            self.clip_model,
            _,
            self.clip_preprocess,
        ) = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="laion2b_s32b_b82k"
        )
        self.clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")

        # initialize sentence transformer
        self.sbert_model = SentenceTransformer("all-mpnet-base-v2", device="cpu")

        # objaverse version and asset dir
        self.objaverse_asset_dir = objaverse_asset_dir

        # initialize generation
        self.retrieval_threshold = 28
        self.object_retriever = ObjathorRetriever(
            clip_model=self.clip_model,
            clip_preprocess=self.clip_preprocess,
            clip_tokenizer=self.clip_tokenizer,
            sbert_model=self.sbert_model,
            retrieval_threshold=self.retrieval_threshold,
        )
        self.floor_generator = FloorPlanGenerator(
            self.clip_model, self.clip_preprocess, self.clip_tokenizer, self.llm
        )
        self.wall_generator = WallGenerator(self.llm)
        self.door_generator = DoorGenerator(
            self.clip_model, self.clip_preprocess, self.clip_tokenizer, self.llm
        )
        self.window_generator = WindowGenerator(self.llm)
        self.object_selector = ObjectSelector(
            object_retriever=self.object_retriever, llm=self.llm
        )
        self.floor_object_generator = FloorObjectGenerator(
            object_retriever=self.object_retriever, llm=self.llm
        )
        self.wall_object_generator = WallObjectGenerator(
            object_retriever=self.object_retriever, llm=self.llm
        )
        self.ceiling_generator = CeilingObjectGenerator(
            object_retriever=self.object_retriever, llm=self.llm
        )
        self.small_object_generator = SmallObjectGenerator(
            object_retriever=self.object_retriever, llm=self.llm
        )

        # additional requirements
        single_room_requirements = "I only need one room"

        if single_room:
            self.additional_requirements_room = single_room_requirements
        else:
            self.additional_requirements_room = "N/A"

        self.additional_requirements_door = "N/A"
        self.additional_requirements_window = (
            "Only one wall of each room should have windows"
        )
        self.additional_requirements_object = "N/A"
        self.additional_requirements_ceiling = "N/A"

    def get_empty_scene(self):
        return compress_json.load("generation/empty_house.json")

    def empty_house(self, scene):
        scene["rooms"] = []
        scene["walls"] = []
        scene["doors"] = []
        scene["windows"] = []
        scene["objects"] = []
        scene["proceduralParameters"]["lights"] = []
        return scene

    def generate_rooms(self, scene, additional_requirements_room, used_assets=[]):
        self.floor_generator.used_assets = used_assets
        rooms = self.floor_generator.generate_rooms(scene, additional_requirements_room)
        scene["rooms"] = rooms
        return scene

    def generate_walls(self, scene):
        wall_height, walls = self.wall_generator.generate_walls(scene)
        scene["wall_height"] = wall_height
        scene["walls"] = walls
        return scene

    def generate_doors(self, scene, additional_requirements_door="N/A", used_assets=[]):
        self.door_generator.used_assets = used_assets

        # generate doors
        (
            raw_doorway_plan,
            doors,
            room_pairs,
            open_room_pairs,
        ) = self.door_generator.generate_doors(scene, additional_requirements_door)
        scene["raw_doorway_plan"] = raw_doorway_plan
        scene["doors"] = doors
        scene["room_pairs"] = room_pairs
        scene["open_room_pairs"] = open_room_pairs

        # update walls
        updated_walls, open_walls = self.wall_generator.update_walls(
            scene["walls"], open_room_pairs
        )
        scene["walls"] = updated_walls
        scene["open_walls"] = open_walls
        return scene

    def generate_windows(
        self,
        scene,
        additional_requirements_window="I want to install windows to only one wall of each room",
        used_assets=[],
    ):
        self.window_generator.used_assets = used_assets
        raw_window_plan, walls, windows = self.window_generator.generate_windows(
            scene, additional_requirements_window
        )
        scene["raw_window_plan"] = raw_window_plan
        scene["windows"] = windows
        scene["walls"] = walls
        return scene

    def select_objects(self, scene, additional_requirements_object, used_assets=[]):
        self.object_selector.used_assets = used_assets
        object_selection_plan, selected_objects = self.object_selector.select_objects(
            scene, additional_requirements_object
        )
        scene["object_selection_plan"] = object_selection_plan
        scene["selected_objects"] = selected_objects
        return scene

    def generate_ceiling_objects(self, scene, additional_requirements_ceiling="N/A"):
        (
            raw_ceiling_plan,
            ceiling_objects,
        ) = self.ceiling_generator.generate_ceiling_objects(
            scene, additional_requirements_ceiling
        )
        scene["ceiling_objects"] = ceiling_objects
        scene["raw_ceiling_plan"] = raw_ceiling_plan
        return scene

    def generate_small_objects(self, scene, used_assets=[]):
        self.small_object_generator.used_assets = used_assets
        controller = self.small_object_generator.start_controller(
            scene, self.objaverse_asset_dir
        )
        event = controller.reset()
        receptacle_ids = [
            obj["objectId"]
            for obj in event.metadata["objects"]
            if obj["receptacle"] and "___" not in obj["objectId"]
        ]
        if "Floor" in receptacle_ids:
            receptacle_ids.remove("Floor")

        try:
            (
                small_objects,
                receptacle2small_objects,
            ) = self.small_object_generator.generate_small_objects(
                scene, controller, receptacle_ids
            )
            scene["small_objects"] = small_objects
            scene["receptacle2small_objects"] = receptacle2small_objects
        except:
            scene["small_objects"] = []
            print("Failed to generate small objects")

        controller.stop()  # stop controller to avoid memory leak
        return scene

    def change_ceiling_material(self, scene):
        first_wall_material = scene["rooms"][0]["wallMaterial"]
        scene["proceduralParameters"]["ceilingMaterial"] = first_wall_material
        return scene

    def generate_scene(
        self,
        scene,
        query: str,
        save_dir: str,
        used_assets=[],
        add_ceiling=False,
        generate_image=True,
        generate_video=False,
        add_time=True,
        use_constraint=True,
        random_selection=False,
        use_milp=False,
    ) -> Tuple[Dict[str, Any], str]:
        # initialize scene
        query = query.replace("_", " ")
        scene["query"] = query

        # empty house
        scene = self.empty_house(scene)

        # generate rooms
        scene = self.generate_rooms(
            scene,
            additional_requirements_room=self.additional_requirements_room,
            used_assets=used_assets,
        )

        # generate walls
        scene = self.generate_walls(scene)

        # generate doors
        scene = self.generate_doors(
            scene,
            additional_requirements_door=self.additional_requirements_door,
            used_assets=used_assets,
        )

        # generate windows
        scene = self.generate_windows(
            scene,
            additional_requirements_window=self.additional_requirements_window,
            used_assets=used_assets,
        )

        # select objects
        self.object_selector.random_selection = random_selection
        scene = self.select_objects(
            scene,
            additional_requirements_object=self.additional_requirements_object,
            used_assets=used_assets,
        )

        # generate floor objects
        self.floor_object_generator.use_milp = use_milp
        scene["floor_objects"] = self.floor_object_generator.generate_objects(
            scene, use_constraint=use_constraint
        )

        # generate wall objects
        scene["wall_objects"] = self.wall_object_generator.generate_wall_objects(
            scene, use_constraint=use_constraint
        )

        # combine floor and wall objects
        scene["objects"] = scene["floor_objects"] + scene["wall_objects"]

        # generate small objects
        scene = self.generate_small_objects(scene, used_assets=used_assets)
        scene["objects"] += scene["small_objects"]

        # generate ceiling objects
        if add_ceiling:
            scene = self.generate_ceiling_objects(
                scene,
                additional_requirements_ceiling=self.additional_requirements_ceiling,
            )
            scene["objects"] += scene["ceiling_objects"]

        # generate lights
        lights = generate_lights(scene)
        scene["proceduralParameters"]["lights"] = lights

        # assign layers
        scene = map_asset2layer(scene)

        # assign skybox
        scene = getSkybox(scene)

        # change ceiling material
        scene = self.change_ceiling_material(scene)

        # create folder
        query_name = query.replace(" ", "_").replace("'", "")[:30]
        create_time = (
            str(datetime.datetime.now())
            .replace(" ", "-")
            .replace(":", "-")
            .replace(".", "-")
        )

        if add_time:
            folder_name = f"{query_name}-{create_time}"  # query name + time
        else:
            folder_name = query_name  # query name only

        save_dir = os.path.abspath(os.path.join(save_dir, folder_name))
        os.makedirs(save_dir, exist_ok=True)
        compress_json.dump(
            scene,
            os.path.join(save_dir, f"{query_name}.json"),
            json_kwargs=dict(indent=4),
        )

        # save top down image
        if generate_image:
            top_image = get_top_down_frame(scene, self.objaverse_asset_dir, 1024, 1024)
            top_image.show()
            top_image.save(os.path.join(save_dir, f"{query_name}.png"))

        # save video
        if generate_video:
            scene["objects"] = (
                scene["floor_objects"] + scene["wall_objects"] + scene["small_objects"]
            )
            final_video = room_video(scene, self.objaverse_asset_dir, 1024, 1024)
            final_video.write_videofile(
                os.path.join(save_dir, f"{query_name}.mp4"), fps=30
            )

        return scene, save_dir

    def generate_variants(
        self,
        query,
        original_scene,
        save_dir=os.path.join(HOLODECK_BASE_DATA_DIR, "scenes"),
        number_of_variants=5,
        used_assets=[],
    ):
        self.object_selector.reuse_selection = (
            False  # force the selector to retrieve different assets
        )

        # create the list of used assets
        used_assets += [
            obj["assetId"]
            for obj in original_scene["objects"]
            + original_scene["windows"]
            + original_scene["doors"]
        ]
        used_assets += [
            room["floorMaterial"]["name"] for room in original_scene["rooms"]
        ]
        used_assets += [wall["material"]["name"] for wall in original_scene["walls"]]
        used_assets = list(set(used_assets))

        variant_scenes = []
        for i in tqdm(range(number_of_variants)):
            variant_scene, _ = self.generate_scene(
                original_scene.copy(),
                query,
                save_dir,
                used_assets,
                generate_image=True,
                generate_video=False,
                add_time=True,
            )
            variant_scenes.append(variant_scene)
            used_assets += [
                obj["assetId"]
                for obj in variant_scene["objects"]
                + variant_scene["windows"]
                + variant_scene["doors"]
            ]
            used_assets += [
                room["floorMaterial"]["name"] for room in variant_scene["rooms"]
            ]
            used_assets += [wall["material"]["name"] for wall in variant_scene["walls"]]
            used_assets = list(set(used_assets))
        return variant_scenes

    def ablate_placement(
        self,
        scene,
        query,
        save_dir,
        used_assets=[],
        add_ceiling=False,
        generate_image=True,
        generate_video=False,
        add_time=True,
        use_constraint=False,
        constraint_type="llm",
    ):
        # place floor objects
        if use_constraint:
            self.floor_object_generator.constraint_type = (
                constraint_type  # ablate the constraint types
            )
        scene["floor_objects"] = self.floor_object_generator.generate_objects(
            scene, use_constraint=use_constraint
        )
        if len(scene["floor_objects"]) == 0:
            print("No object is placed, skip this scene")
            return None  # if no object is placed, return None
        # place wall objects
        if use_constraint:
            self.wall_object_generator.constraint_type = constraint_type
        scene["wall_objects"] = self.wall_object_generator.generate_wall_objects(
            scene, use_constraint=use_constraint
        )

        # combine floor and wall objects
        scene["objects"] = scene["floor_objects"] + scene["wall_objects"]

        # generate small objects
        scene = self.generate_small_objects(scene, used_assets=used_assets)
        scene["objects"] += scene["small_objects"]

        # assign layers
        scene = map_asset2layer(scene)

        # take the first 30 characters of the query as the folder name
        query_name = query.replace(" ", "_").replace("'", "")[:30]
        create_time = (
            str(datetime.datetime.now())
            .replace(" ", "-")
            .replace(":", "-")
            .replace(".", "-")
        )

        if add_time:
            folder_name = f"{query_name}-{create_time}"  # query name + time
        else:
            folder_name = query_name  # query name only

        os.makedirs(f"{save_dir}/{folder_name}", exist_ok=True)
        compress_json.dump(
            scene,
            f"{save_dir}/{folder_name}/{query_name}.json",
            json_kwargs=dict(indent=4),
        )

        # save top down image
        if generate_image:
            top_image = get_top_down_frame(scene, self.objaverse_asset_dir, 1024, 1024)
            top_image.show()
            top_image.save(f"{save_dir}/{folder_name}/{query_name}.png")

        return scene
