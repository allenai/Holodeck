<h2 align="center">
    <img src="https://yueyang1996.github.io/images/logo.png" width="200px"/><br/>
    Language Guided Generation of 3D Embodied AI Environments<br>
</h2>

<h5 align="center">
<img src="https://yueyang1996.github.io/images/office_example.png" width="800px"/><br/>
</h5>

<h4 align="center">
  <a href="tbd">Paper</i></a> | <a href="https://yueyang1996.github.io/holodeck/">Project Page</i></a>
</h4>

## Installation
Use the following commands to install the required dependencies.
```
conda create --name holodeck python=3.9.16
conda activate holodeck
pip install -r requirements.txt
pip install --extra-index-url https://ai2thor-pypi.allenai.org ai2thor==0+6f165fdaf3cf2d03728f931f39261d14a67414d0
```

## Data
Download the data from [here](https://drive.google.com/file/d/1MQbFbNfTz94x8Pxfkgbohz4l46O5e3G1/view?usp=sharing) and extract it to the `data/` folder, or use the following command:
```
FILE_ID=1MQbFbNfTz94x8Pxfkgbohz4l46O5e3G1
CONFIRM=$(curl -sc /tmp/gcookie "https://drive.google.com/uc?export=download&id=${FILE_ID}" | grep -o 'confirm=[^&]*' | sed 's/confirm=//')
wget --load-cookies /tmp/gcookie "https://drive.google.com/uc?export=download&confirm=${CONFIRM}&id=${FILE_ID}" -O data.zip && rm -rf /tmp/gcookie
unzip data.zip
```

## Usage
You can use the following command to generate a new environment.
```
python main.py --query "a living room" --openai_api_key <OPENAI_API_KEY>
```
To be noticed, our system is using `gpt-4-1106-preview`, please make sure you have access to it.


## Citation
Please cite the following paper if you use this code in your work.

```bibtex
@inproceedings{holodeck,
  title={Language Guided Generation of 3D Embodied AI Environments}
}
```
<br />

<a href="//prior.allenai.org">
<p align="center"><img width="100%" src="https://raw.githubusercontent.com/allenai/ai2thor/main/doc/static/ai2-prior.svg" /></p>
</a>