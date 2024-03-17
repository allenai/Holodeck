import copy
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser
from ai2thor.controller import Controller
from ai2thor.hooks.procedural_asset_hook import ProceduralAssetHookRunner
from moviepy.editor import TextClip, CompositeVideoClip, concatenate_videoclips, ImageSequenceClip


def all_edges_white(img):
    # Define a white pixel
    white = [255, 255, 255]

    # Check top edge
    if not np.all(np.all(img[0, :] == white, axis=-1)):
        return False
    # Check bottom edge
    if not np.all(np.all(img[-1, :] == white, axis=-1)):
        return False
    # Check left edge
    if not np.all(np.all(img[:, 0] == white, axis=-1)):
        return False
    # Check right edge
    if not np.all(np.all(img[:, -1] == white, axis=-1)):
        return False

    # If all the conditions met
    return True


def get_top_down_frame(scene, objaverse_asset_dir, width=1024, height=1024):
    controller = Controller(
        agentMode="default",
        makeAgentsVisible=False,
        visibilityDistance=1.5,
        scene=scene,
        width=width,
        height=height,
        fieldOfView=90,
        action_hook_runner=ProceduralAssetHookRunner(
            asset_directory=objaverse_asset_dir,
            asset_symlink=True,
            verbose=True,
        )
    )

    # Setup the top-down camera
    event = controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
    pose = copy.deepcopy(event.metadata["actionReturn"])

    bounds = event.metadata["sceneBounds"]["size"]

    pose["fieldOfView"] = 60
    pose["position"]["y"] = bounds["y"]
    del pose["orthographicSize"]

    try: wall_height = wall_height = max([point["y"] for point in scene["walls"][0]["polygon"]])
    except: wall_height = 2.5

    for i in range(20):    
        pose["orthographic"] = False
        
        pose["farClippingPlane"] = pose["position"]["y"] + 10
        pose["nearClippingPlane"] = pose["position"]["y"] - wall_height
        
        # add the camera to the scene
        event = controller.step(
            action="AddThirdPartyCamera",
            **pose,
            skyboxColor="white",
            raise_for_failure=True,
        )
        top_down_frame = event.third_party_camera_frames[-1]

        # check if the edge of the frame is white
        if all_edges_white(top_down_frame): break
        
        pose["position"]["y"] += 0.75

    controller.stop()
    image = Image.fromarray(top_down_frame)

    return image


def get_top_down_frame_ithor(scene, objaverse_asset_dir, width=1024, height=1024):
    controller = Controller(
        agentMode="default",
        makeAgentsVisible=False,
        visibilityDistance=1.5,
        scene=scene,
        width=width,
        height=height,
        fieldOfView=90,
        action_hook_runner=ProceduralAssetHookRunner(
            asset_directory=objaverse_asset_dir,
            asset_symlink=True,
            verbose=True,
        )
    )
    
    controller.reset(scene)
    
    event = controller.step(action="GetMapViewCameraProperties")
    pose = copy.deepcopy(event.metadata["actionReturn"])
    
    event = controller.step(
            action="AddThirdPartyCamera",
            **pose,
            skyboxColor="white",
            raise_for_failure=True,
        )

    controller.stop()

    top_down_frame = event.third_party_camera_frames[0]

    return Image.fromarray(top_down_frame)


def main(save_path):
    scene = json.load(open(save_path + f"scene.json", "r"))
    image = get_top_down_frame(scene)
    image.save(f"test1.png")
    with open(save_path + f"scene.json", "w") as f:
        json.dump(scene, f, indent=4)


def visualize_asset(asset_id, version):
    empty_house = json.load(open("empty_house.json", "r"))
    empty_house["objects"] = [{
            "assetId": asset_id,
            "id": "test_asset",
            "kinematic": True,
            "position": {
                "x": 0,
                "y": 0,
                "z": 0
            },
            "rotation": {
                "x": 0,
                "y": 0,
                "z": 0
            },
            "material": None
        }]
    image = get_top_down_frame(empty_house, version)
    image.show()


def get_room_images(scene, objaverse_asset_dir, width=1024, height=1024):
    controller = Controller(
        agentMode="default",
        makeAgentsVisible=False,
        visibilityDistance=1.5,
        scene=scene,
        width=width,
        height=height,
        fieldOfView=135,
        action_hook_runner=ProceduralAssetHookRunner(
            asset_directory=objaverse_asset_dir,
            asset_symlink=True,
            verbose=True,
        ),
    )

    wall_height = max([point["y"] for point in scene["walls"][0]["polygon"]])

    room_images = {}
    for room in scene["rooms"]:
        room_name = room["roomType"]
        camera_height = wall_height - 0.2

        room_vertices = [[point["x"], point["z"]] for point in room['floorPolygon']]

        room_center = np.mean(room_vertices, axis=0)
        floor_center = np.array([room_center[0], 0, room_center[1]])
        camera_center = np.array([room_center[0], camera_height, room_center[1]])
        corners = np.array([[point[0], camera_height, point[1]] for point in room_vertices])
        farest_corner = np.argmax(np.linalg.norm(corners - camera_center, axis=1))

        vector_1 = floor_center - camera_center
        vector_2 = farest_corner - camera_center
        x_angle = 90 - np.arccos(np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))) * 180 / np.pi

        if not controller.last_event.third_party_camera_frames:
            controller.step(
                action="AddThirdPartyCamera",
                position=dict(x=camera_center[0], y=camera_center[1], z=camera_center[2]),
                rotation=dict(x=0, y=0, z=0),
            )
        
        images = []
        for angle in tqdm(range(0, 360, 90)):
            controller.step(
                action="UpdateThirdPartyCamera",
                rotation=dict(x=x_angle, y=angle+45, z=0),
                position=dict(x=camera_center[0], y=camera_center[1], z=camera_center[2]),
            )
            images.append(Image.fromarray(controller.last_event.third_party_camera_frames[0]))
        
        room_images[room_name] = images
    
    controller.stop()
    return room_images


def ithor_video(scene, objaverse_asset_dir, width, height, scene_type):
    controller = Controller(
        agentMode="default",
        makeAgentsVisible=False,
        visibilityDistance=2,
        scene=scene,
        width=width,
        height=height,
        fieldOfView=90,
        action_hook_runner=ProceduralAssetHookRunner(
            asset_directory=objaverse_asset_dir,
            asset_symlink=True,
            verbose=True,
        ),
    )

    event = controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
    pose = copy.deepcopy(event.metadata["actionReturn"])

    wall_height = 2.5
    camera_height = wall_height - 0.2

    if not controller.last_event.third_party_camera_frames:
        controller.step(
            action="AddThirdPartyCamera",
            position=dict(x=pose["position"]["x"], y=camera_height, z=pose["position"]["z"]),
            rotation=dict(x=0, y=0, z=0),
        )
    
    images = []

    for angle in tqdm(range(0, 360, 1)):
        controller.step(
            action="UpdateThirdPartyCamera",
            rotation=dict(x=45, y=angle, z=0),
            position=dict(x=pose["position"]["x"], y=camera_height, z=pose["position"]["z"]),
        )
        images.append(controller.last_event.third_party_camera_frames[0])

    imsn = ImageSequenceClip(images, fps=30)

    # Create text clips
    txt_clip_query = (TextClip(f"Query: {scene_type}", fontsize=30, color='white', font='Arial-Bold')
                        .set_pos(('center', 'top')).set_duration(imsn.duration))
    txt_clip_room = (TextClip(f"Room Type: {scene_type}", fontsize=30, color='white', font='Arial-Bold')
                        .set_pos(('center', 'bottom')).set_duration(imsn.duration))

    # Overlay the text clip on the first video clip
    video = CompositeVideoClip([imsn, txt_clip_query, txt_clip_room])

    controller.stop()

    return video


def room_video(scene, objaverse_asset_dir, width, height):
    def add_line_breaks(text, max_line_length):
        words = text.split(' ')
        lines = []
        current_line = []

        for word in words:
            if len(' '.join(current_line + [word])) <= max_line_length:
                current_line.append(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]

        lines.append(' '.join(current_line))

        return '\n'.join(lines)

    """Saves a top-down video of the house."""
    controller = Controller(
        agentMode="default",
        makeAgentsVisible=False,
        visibilityDistance=2,
        scene=scene,
        width=width,
        height=height,
        fieldOfView=90,
        action_hook_runner=ProceduralAssetHookRunner(
            asset_directory=objaverse_asset_dir,
            asset_symlink=True,
            verbose=True,
        ),
    )
    
    try: query = scene["query"]
    except: query = scene["rooms"][0]['roomType']

    wall_height = max([point["y"] for point in scene["walls"][0]["polygon"]])

    text_query = add_line_breaks(query, 60)
    videos = []
    for room in scene["rooms"]:
        room_name = room["roomType"]
        camera_height = wall_height - 0.2
        print("camera height: ", camera_height)

        room_vertices = [[point["x"], point["z"]] for point in room['floorPolygon']]

        room_center = np.mean(room_vertices, axis=0)
        floor_center = np.array([room_center[0], 0, room_center[1]])
        camera_center = np.array([room_center[0], camera_height, room_center[1]])
        corners = np.array([[point["x"], point["y"], point["z"]] for point in room['floorPolygon']])
        farest_corner = corners[np.argmax(np.linalg.norm(corners - camera_center, axis=1))]

        vector_1 = floor_center - camera_center
        vector_2 = farest_corner - camera_center
        x_angle = 90 - np.arccos(np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))) * 180 / np.pi

        images = []
        if not controller.last_event.third_party_camera_frames:
            controller.step(
                action="AddThirdPartyCamera",
                position=dict(x=camera_center[0], y=camera_center[1], z=camera_center[2]),
                rotation=dict(x=0, y=0, z=0),
            )

        for angle in tqdm(range(0, 360, 1)):
            controller.step(
                action="UpdateThirdPartyCamera",
                rotation=dict(x=x_angle, y=angle, z=0),
                position=dict(x=camera_center[0], y=camera_center[1], z=camera_center[2]),
            )
            images.append(controller.last_event.third_party_camera_frames[0])

        imsn = ImageSequenceClip(images, fps=30)
        
        # Create text clips
        txt_clip_query = (TextClip(f"Query: {text_query}", fontsize=30, color='white', font='Arial-Bold')
                          .set_pos(('center', 'top')).set_duration(imsn.duration))
        txt_clip_room = (TextClip(f"Room Type: {room_name}", fontsize=30, color='white', font='Arial-Bold')
                         .set_pos(('center', 'bottom')).set_duration(imsn.duration))

        # Overlay the text clip on the first video clip
        video = CompositeVideoClip([imsn, txt_clip_query, txt_clip_room])

        # Add this room's video to the list
        videos.append(video)

    # Concatenate all room videos into one final video
    final_video = concatenate_videoclips(videos)
    controller.stop()

    return final_video


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mode", help = "Mode to run (top_down_frame, room_video, room_image).", default = "top_down_frame")
    parser.add_argument("--objaverse_asset_dir", help = "Directory to load assets from.", default = "./data/objaverse_holodeck/09_23_combine_scale/processed_2023_09_23_combine_scale")
    parser.add_argument("--scene", help = "Scene to load.", default = "data/scenes/a_living_room/a_living_room.json")

    args = parser.parse_args()
    scene = json.load(open(args.scene, "r"))

    if "query" not in scene: scene["query"] = args.scene.split("/")[-1].split(".")[0]

    if args.mode == "top_down_frame":
        image = get_top_down_frame(scene, args.objaverse_asset_dir)
        image.show()
    
    elif args.mode == "room_video":
        video = room_video(scene, args.objaverse_asset_dir, 1024, 1024)
        video.write_videofile(args.scene.replace(".json", ".mp4"), fps=30)
    
    elif args.mode == "room_image":
        room_images = get_room_images(scene, args.objaverse_asset_dir, 1024, 1024)
        save_folder = "/".join(args.scene.split("/")[:-1])
        for room_name, images in room_images.items():
            for i, image in enumerate(images):
                image.save(f"{save_folder}/{room_name}_{i}.png")