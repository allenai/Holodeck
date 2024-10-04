import os
from pathlib import Path

import objathor.dataset
from ai2thor.hooks.procedural_asset_hook import WebProceduralAssetHookRunner

ABS_PATH_OF_HOLODECK = os.path.abspath(os.path.dirname(Path(__file__)))

# ASSETS_VERSION = os.environ.get("ASSETS_VERSION", "2023_09_23")
ASSETS_VERSION = os.environ.get("ASSETS_VERSION", "2024_08_16")
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

THOR_COMMIT_ID = "455cf72a1c8e0759a452422f2128fbc93a3cb06b"

BASE_URL = objathor.dataset.DatasetSaveConfig(
    VERSION=ASSETS_VERSION
).VERSIONED_BUCKET_URL

PROCEDURAL_ASSET_HOOK_RUNNER = WebProceduralAssetHookRunner(
    asset_directory=OBJATHOR_ASSETS_DIR,
    base_url=BASE_URL.strip("/") + "/assets",
    target_dir=ASSETS_VERSION,
    asset_symlink=True,
    verbose=True,
)

if ASSETS_VERSION == "2024_08_16":
    os.makedirs(OBJATHOR_ASSETS_DIR, exist_ok=True)

# LLM_MODEL_NAME = "gpt-4-1106-preview"
LLM_MODEL_NAME = "gpt-4o-2024-08-06"
SMALL_LLM_MODEL_NAME = "gpt-4o-mini-2024-07-18"

DEBUGGING = os.environ.get("DEBUGGING", "0").lower() in ["1", "true", "True", "t", "T"]

MULTIPROCESSING = os.environ.get("MULTIPROCESSING", "1").lower() in [
    "1",
    "true",
    "t",
]
