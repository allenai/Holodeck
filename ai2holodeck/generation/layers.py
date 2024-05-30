def get_room2layer(room_pairs, open_room_pairs):
    # Create an adjacency list
    adjacency_list = {}
    for pair in room_pairs:
        if pair[0] not in adjacency_list:
            adjacency_list[pair[0]] = []
        if pair[1] not in adjacency_list:
            adjacency_list[pair[1]] = []
        adjacency_list[pair[0]].append(pair[1])
        adjacency_list[pair[1]].append(pair[0])

    # Remove open room pairs from the adjacency list
    for pair in open_room_pairs:
        try:
            adjacency_list[pair[0]].remove(pair[1])
            adjacency_list[pair[1]].remove(pair[0])
        except:
            print(adjacency_list)
            continue

    # Initialize colors (-1 means no color assigned yet)
    colors = {room: -1 for room in adjacency_list.keys()}

    # Define the color assignment function
    def assign_color(room, color):
        for neighbor in adjacency_list[room]:
            if colors[neighbor] == color:
                return False
        return True

    # Define the coloring function
    def color_rooms(room):
        if room == len(adjacency_list.keys()):
            return True
        for color in range(4):  # Use color range 0-3
            if assign_color(list(adjacency_list.keys())[room], f"Procedural{color}"):
                colors[list(adjacency_list.keys())[room]] = f"Procedural{color}"
                if color_rooms(room + 1):
                    return True
                colors[list(adjacency_list.keys())[room]] = -1
        return False

    # Start the coloring
    if not color_rooms(0):
        return None

    return colors


def map_asset2layer(scene):
    room2layer = get_room2layer(scene["room_pairs"], scene["open_room_pairs"])
    all_layers = ["Procedural0", "Procedural1", "Procedural2", "Procedural3"]

    if len(scene["rooms"]) == 1:
        print("Only one room in the scene. Assigning the room to Procedural0.")
        room2layer = {scene["rooms"][0]["id"]: "Procedural0"}

    # Check if all rooms are assigned a layer
    for room in scene["rooms"]:
        if room["id"] not in room2layer:
            room2layer[room["id"]] = "Procedural0"

    # Assign layer to each room
    for room in scene["rooms"]:
        room["layer"] = room2layer[room["id"]]

    # Assign layer to each wall
    for wall in scene["walls"]:
        wall["layer"] = room2layer[wall["roomId"]]

    # Assign layer to each object
    # TODO: consider small children objects
    for obj in scene["objects"]:
        obj["layer"] = room2layer[obj["roomId"]]

    # Assign layer to each window
    for window in scene["windows"]:
        window["layer"] = room2layer[window["roomId"]]

    # Assign layer to each light
    for light in scene["proceduralParameters"]["lights"]:
        try:
            light["layer"] = room2layer[light["roomId"]]
        except:
            continue

        light["cullingMaskOff"] = [
            layer for layer in all_layers if layer != light["layer"]
        ]

    return scene


if __name__ == "__main__":
    room_pairs = [
        ("Living Room", "Bedroom"),
        ("Living Room", "Kitchen"),
        ("Kitchen", "Bathroom"),
        ("Bedroom", "Bathroom"),
    ]
    open_room_pairs = [("Living Room", "Kitchen"), ("Living Room", "Bedroom")]
    room2layer = get_room2layer(room_pairs, open_room_pairs)
