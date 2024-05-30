import os
from pathlib import Path

ABS_PATH_OF_HOLODECK = os.path.abspath(os.path.dirname(Path(__file__)))

ASSETS_VERSION = os.environ.get("ASSETS_VERSION", "2023_09_23")
HD_BASE_VERSION = os.environ.get("HD_BASE_VERSION", "2023_09_23")

OBJATHOR_ASSETS_BASE_DIR = os.environ.get(
    "OBJATHOR_ASSETS_BASE_DIR", os.path.expanduser(f"~/.objathor-assets")
)

OBJATHOR_VERSIONED_DIR = os.path.join(OBJATHOR_ASSETS_BASE_DIR, ASSETS_VERSION)
OBJATHOR_ASSETS_DIR = os.path.join(OBJATHOR_VERSIONED_DIR, "assets")
OBJATHOR_FEATURES_DIR = os.path.join(OBJATHOR_VERSIONED_DIR, "features")
OBJATHOR_ANNOTATIONS_PATH = os.path.join(OBJATHOR_VERSIONED_DIR, "annotations.json.gz")

HOLODECK_BASE_DATA_DIR = os.path.join(
    OBJATHOR_ASSETS_BASE_DIR, "holodeck", HD_BASE_VERSION
)

HOLODECK_THOR_FEATURES_DIR = os.path.join(HOLODECK_BASE_DATA_DIR, "thor_object_data")
HOLODECK_THOR_ANNOTATIONS_PATH = os.path.join(
    HOLODECK_BASE_DATA_DIR, "thor_object_data", "annotations.json.gz"
)

if ASSETS_VERSION > "2023_09_23":
    THOR_COMMIT_ID = "8524eadda94df0ab2dbb2ef5a577e4d37c712897"
else:
    THOR_COMMIT_ID = "3213d486cd09bcbafce33561997355983bdf8d1a"

LLM_MODEL_NAME = "gpt-4-1106-preview"

DEBUGGING = os.environ.get("DEBUGGING", "0").lower() in ["1", "true", "True", "t", "T"]
