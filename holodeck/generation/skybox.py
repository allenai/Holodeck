import random

from procthor.utils.types import RGB, Vector3

skyboxes = ["Sky1", "Sky2", "SkyAlbany", "SkyAlbanyHill", "SkyDalyCity", "SkyEmeryville", "SkyGarden", "SkyTropical",
            "SkyGasworks", "SkyMosconeCenter", "SkyMountain", "SkyOakland", "SkySeaStacks", "SkySFCityHall", "Sky2Dusk",
            "SkySFDowntown", "SkySFGarden", "SkySnow", "SkyNeighborhood", "SkySouthLakeUnion", "SkySunset", "SkyTreeFarm"]
# timeOfDays = ["Midday", "GoldenHour", "BlueHour", "Midnight"]
timeOfDays = ["Midday", "GoldenHour", "BlueHour"]

def getSkybox(scene):
    skybox = random.choice(skyboxes)
    time_of_day = random.choice(timeOfDays)

    scene["proceduralParameters"]["skyboxId"] = skybox
    lights = scene["proceduralParameters"]["lights"]
    directional_light = lights[0]
    point_lights = lights[1:]

    if time_of_day == "Midday":
        directional_light["intensity"] = 1
        directional_light["rgb"] = RGB(r=1.0, g=1.0, b=1.0)
        directional_light["rotation"] = Vector3(x=66, y=75, z=0)
        # TODO: This will make room too dark
        # for point_light in point_lights:
        #     point_light["intensity"] = 0.45

    elif time_of_day == "GoldenHour":
        directional_light["intensity"] = 1
        directional_light["rgb"] = RGB(r=1.0, g=0.694, b=0.78)
        directional_light["rotation"] = Vector3(x=6, y=-166, z=0)

    elif time_of_day == "BlueHour":
        directional_light["intensity"] = 0.5
        directional_light["rgb"] = RGB(r=0.638, g=0.843, b=1.0)
        directional_light["rotation"] = Vector3(x=82, y=-30, z=0)

    elif time_of_day == "Midnight":
        directional_light["intensity"] = 0.3
        directional_light["rgb"] = RGB(r=0.93, g=0.965, b=1.0)
        directional_light["rotation"] = Vector3(x=41, y=-50, z=0)

    return scene