from shapely import Polygon
from procthor.utils.types import RGB, Light, LightShadow, Vector3

def generate_lights(scene):
    lights = [
        Light(
            id="DirectionalLight",
            position=Vector3(x=0.84, y=0.1855, z=-1.09),
            rotation=Vector3(x=43.375, y=-3.902, z=-63.618),
            shadow=LightShadow(
                type="Soft",
                strength=1,
                normalBias=0,
                bias=0,
                nearPlane=0.2,
                resolution="FromQualitySettings",
            ),
            type="directional",
            intensity=0.35,
            indirectMultiplier=1.0,
            rgb=RGB(r=1.0, g=1.0, b=1.0),
        )
    ]

    for room in scene["rooms"]:
        room_id = room["id"]
        floor_polygon = Polygon(room["vertices"])
        x = floor_polygon.centroid.x
        z = floor_polygon.centroid.y

        light_height = scene["wall_height"] - 0.2
        try:
            for object in scene["ceiling_objects"]:
                if object["roomId"] == room_id: light_height = object["position"]["y"] - 0.2
        except:
            light_height = scene["wall_height"] - 0.2
                    

        lights.append(
            Light(
                id=f"light|{room_id}",
                type="point",
                position=Vector3(x=x, y=light_height, z=z),
                intensity=0.75,
                range=15,
                rgb=RGB(r=1.0, g=0.855, b=0.722),
                shadow=LightShadow(
                    type="Soft",
                    strength=1,
                    normalBias=0,
                    bias=0.05,
                    nearPlane=0.2,
                    resolution="FromQualitySettings",
                ),
                roomId=room_id
            )
        )

    return lights