<h2 align="center">
    <img src="https://yueyang1996.github.io/images/logo.png" width="200px"/><br/>
    Language Guided Generation of 3D Embodied AI Environments<br>
</h2>

<h5 align="center">
<img src="https://yueyang1996.github.io/images/office_example.png" width="800px"/><br/>
</h5>

<h4 align="center">
  <a href="https://arxiv.org/abs/2312.09067"><i>Paper</i></a> | <a href="https://yueyang1996.github.io/holodeck/"><i>Project Page</i></a>
</h4>

## Requirements
Holodeck is based on [AI2-THOR](https://ai2thor.allenai.org/ithor/documentation/#requirements), and we currently support macOS 10.9+ or Ubuntu 14.04+.

**New Feature**: To add ANY new assets to AI2-THOR, please check the [objathor repo](https://github.com/allenai/objathor)!

**Note:** To yield better layouts, use `DFS` as the solver. If you pull the repo before `12/28/2023`, you must set the [argument](https://github.com/allenai/Holodeck/blob/386b0a868def29175436dc3b1ed85b6309eb3cad/main.py#L78) `--use_milp` to `False` to use `DFS`.

## Installation
After cloning the repo, you can install the required dependencies using the following commands:
```
conda create --name holodeck python=3.10
conda activate holodeck
pip install -r requirements.txt
pip install --extra-index-url https://ai2thor-pypi.allenai.org ai2thor==0+8524eadda94df0ab2dbb2ef5a577e4d37c712897
```

## Data
Download the data by running the following commands:
```bash
python -m objathor.dataset.download_holodeck_base_data --version 2023_09_23
python -m objathor.dataset.download_assets --version 2023_09_23
python -m objathor.dataset.download_annotations --version 2023_09_23
python -m objathor.dataset.download_features --version 2023_09_23
```
by default these will save to `~/.objathor-assets/...`, you can change this director by specifying the `--path` argument.  If you change the `--path`, you'll need to set the `OBJAVERSE_ASSETS_DIR` environment variable to the path where the assets are stored when you use Holodeck.

## Usage
You can use the following command to generate a new environment.
```
python holodeck/main.py --query "a living room" --openai_api_key <OPENAI_API_KEY>
```
Our system uses `gpt-4-1106-preview`, **so please ensure you have access to it.**

**Note:** To yield better layouts, use `DFS` as the solver. If you pull the repo before `12/28/2023`, you must set the [argument](https://github.com/allenai/Holodeck/blob/386b0a868def29175436dc3b1ed85b6309eb3cad/main.py#L78) `--use_milp` to `False` to use `DFS`.

## Load the scene in Unity
1. Install [Unity](https://unity.com/download) and select the editor version `2020.3.25f1`.
2. Clone [AI2-THOR repository](https://github.com/allenai/ai2thor) and switch to the new_cam_adjust branch.
```
git clone https://github.com/allenai/ai2thor.git
git checkout 6f165fdaf3cf2d03728f931f39261d14a67414d0
```
3. Reinstall some packages:
```
pip uninstall Werkzeug
pip uninstall Flask
pip install Werkzeug==2.0.1
pip install Flask==2.0.1
```
3. Load `ai2thor/unity` as project in Unity and open `ai2thor/unity/Assets/Scenes/Procedural/Procedural.unity`.
4. In the terminal, run [this python script](connect_to_unity.py):
```
python connect_to_unity --scene <SCENE_JSON_FILE_PATH>
```
5. Press the play button (the triangle) in Unity to view the scene.

## Citation
Please cite the following paper if you use this code in your work.

```bibtex
@InProceedings{Yang_2024_CVPR,
    author    = {Yang, Yue and Sun, Fan-Yun and Weihs, Luca and VanderBilt, Eli and Herrasti, Alvaro and Han, Winson and Wu, Jiajun and Haber, Nick and Krishna, Ranjay and Liu, Lingjie and Callison-Burch, Chris and Yatskar, Mark and Kembhavi, Aniruddha and Clark, Christopher},
    title     = {Holodeck: Language Guided Generation of 3D Embodied AI Environments},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {16227-16237}
}
```
<br />

<a href="//prior.allenai.org">
<p align="center"><img width="100%" src="https://raw.githubusercontent.com/allenai/ai2thor/main/doc/static/ai2-prior.svg" /></p>
</a>
