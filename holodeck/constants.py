import os
from pathlib import Path

ABS_PATH_OF_HOLODECK = os.path.abspath(os.path.dirname(Path(__file__)))

VERSION = "2023_09_23"

OBJATHOR_VERSIONED_DIR = os.path.expanduser(f"~/.objathor-assets/{VERSION}")
OBJATHOR_ASSETS_DIR = os.path.join(OBJATHOR_VERSIONED_DIR, "assets")
OBJATHOR_FEATURES_DIR = os.path.join(OBJATHOR_VERSIONED_DIR, "features")
OBJATHOR_ANNOTATIONS_PATH = os.path.join(OBJATHOR_VERSIONED_DIR, "annotations.json.gz")

HOLODECK_BASE_DATA_DIR = os.path.expanduser(f"~/.objathor-assets/holodeck/{VERSION}")

HOLODECK_THOR_FEATURES_DIR = os.path.join(HOLODECK_BASE_DATA_DIR, "thor_object_data")
HOLODECK_THOR_ANNOTATIONS_PATH = os.path.join(
    HOLODECK_BASE_DATA_DIR, "thor_object_data", "annotations.json.gz"
)

THOR_COMMIT_ID = "3213d486cd09bcbafce33561997355983bdf8d1a"
