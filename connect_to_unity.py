import json
import ai2thor
from ai2thor.controller import Controller
from ai2thor.hooks.procedural_asset_hook import ProceduralAssetHookRunner
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--scene", help = "the directory of the scene to be generated", default = "./data/scenes/a_living_room/a_living_room.json")
parser.add_argument("--asset_dir", help = "the directory of the assets to be used", default = "./data/objaverse_holodeck/09_23_combine_scale/processed_2023_09_23_combine_scale")
args = parser.parse_args()

scene = json.load(open(args.scene, "r"))

controller = Controller(
        start_unity=False,
        port=8200,
        scene="Procedural",
        gridSize=0.25,
        width=300,
        height=300,
        server_class=ai2thor.wsgi_server.WsgiServer,
        makeAgentsVisible=False,
        visibilityScheme='Distance',
        action_hook_runner=ProceduralAssetHookRunner(
        asset_directory=args.asset_dir,
        asset_symlink=True,
        verbose=True,
        )
    )


controller.step(action="CreateHouse", house=scene)
print("controller reset")