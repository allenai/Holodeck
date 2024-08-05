from typing import List, TypedDict, Dict, Literal, Union, Any, Optional, NamedTuple


class ObjectsOnTopDict(TypedDict):
    object_name: str
    quantity: int
    variance_type: str
    importance: Union[int, float]


class FloorOrWallObjectDict(TypedDict):
    object_name: str
    description: str
    location: Literal["floor", "wall"]
    size: Optional[List[Union[float, int]]]
    quantity: int
    variance_type: Literal["same", "varied"]
    importance: Union[int, float]
    objects_on_top: List[ObjectsOnTopDict]


class ObjectPlanForRoomDict(Dict[str, FloorOrWallObjectDict]):
    pass  # Maps object names to object plans


class ObjectPlanForSceneDict(Dict[str, ObjectPlanForRoomDict]):
    pass  # Maps room_ids to object plans


def _normalize_attribute_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    return {key.strip().lower().replace(" ", "_"): value for key, value in d.items()}


def _recursively_normalize_attribute_keys(obj: Any) -> Any:
    if isinstance(obj, Dict):
        return {
            key.strip()
            .lower()
            .replace(" ", "_"): _recursively_normalize_attribute_keys(value)
            for key, value in obj.items()
        }
    elif isinstance(obj, List):
        return [_recursively_normalize_attribute_keys(value) for value in obj]
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    else:
        print(
            f"Unexpected type {type(obj)} in {obj} while normalizing attribute keys."
            f" Returning the object as is."
        )
        return obj


def objects_on_top_from_dict(obj: Dict[str, Any]) -> Optional[ObjectsOnTopDict]:
    try:
        _normalize_attribute_keys(obj)
        object_name = obj["object_name"]
        quantity = int(obj["quantity"])
        variance_type = obj.get("variance_type", "same")
        if variance_type not in ["same", "varied"]:
            obj["variance_type"] = (
                "same" if not variance_type.startswith("v") else "varied"
            )
        importance = float(obj.get("importance", 0))
    except (KeyError, ValueError):
        return None

    return {
        "object_name": object_name,
        "quantity": quantity,
        "variance_type": variance_type,
        "importance": importance,
    }


def floor_or_wall_object_from_dict(
    obj: Dict[str, Any]
) -> Optional[FloorOrWallObjectDict]:
    try:
        _normalize_attribute_keys(obj)

        object_name = obj.get("object_name")
        description = obj["description"]
        location = obj.get("location", "floor")

        size = obj.get("size", None)
        if (
            not isinstance(size, List)
            or len(size) != 3
            or not all(isinstance(i, int) for i in size)
        ):
            obj["size"] = None

        quantity = int(obj["quantity"])

        variance_type = obj.get("variance_type", "same")
        if variance_type not in ["same", "varied"]:
            obj["variance_type"] = (
                "same" if not variance_type.startswith("v") else "varied"
            )

        importance = float(obj.get("importance", 0))
        objects_on_top = [
            objects_on_top_from_dict(obj) for obj in obj.get("objects_on_top", [])
        ]
        objects_on_top = [obj for obj in objects_on_top if obj is not None]
    except (KeyError, ValueError):
        return None

    return {
        "object_name": object_name,
        "description": description,
        "location": location,
        "size": size,
        "quantity": quantity,
        "variance_type": variance_type,
        "importance": importance,
        "objects_on_top": objects_on_top,
    }


def object_plan_from_dict(obj: Dict[str, Dict[str, Any]]) -> ObjectPlanForRoomDict:

    opd = {key: floor_or_wall_object_from_dict(value) for key, value in obj.items()}
    return ObjectPlanForRoomDict(
        (k, {**v, "object_name": k}) for k, v in opd.items() if v is not None
    )


class SelectedSmallObject(NamedTuple):
    object_name: str
    asset_id: str
    importance: float
    size: float


class PlacedSmallObject(TypedDict):
    assetId: str
    id: str
    kinematic: bool
    position: Dict[str, float]
    rotation: Dict[str, float]
    material: str
    roomId: str


class HolodeckScenePlanDict(TypedDict):
    object_selection_plan: ObjectPlanForSceneDict
    receptacle_id_to_small_objects: Dict[str, PlacedSmallObject]
    small_objects: List[PlacedSmallObject]


class SmallObjectPlan(TypedDict):
    object_name: str
    quantity: int
    variance_type: str
    importance: Union[int, float]

    room_name: str  # Name of the room this object is in, may not exist
    parent: str  # Object name of the parent receptacle, may not exist
