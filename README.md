# ComfyUI-Tripo
This extension integrates Tripo into ComfyUI, allowing users to generate 3D models from text prompts or images directly within the ComfyUI interface.

## ChangeLog
- 20250619: update to the newest api; add more nodes
- 20250331: use tripo3d package
- 20250224: remove glbviewer
- 20250201: adapt for new api; use preview3D for viewing models
- 20241111: adapt for new api
- 20241014: support convert
- 20240913: support model_version v2.0-20240919

## Features
- Generate 3D models from text prompts
- Generate 3D models from images
- Animate 3d models
- Convert format and retopologize

## Installation
### [method1] From Source
- Clone or download this repository into your `ComfyUI/custom_nodes/` directory.
- Install the required dependencies by running `pip install -r requirements.txt`.

### [method2] From [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)
If you have ComfyUI-Manager, you can simply search "Tripo for ComfyUI" from `Custom Nodes Manager` and install these custom nodes 

### [method3] From [Comfy Register](https://registry.comfy.org/) using [comfy-cli](https://github.com/Comfy-Org/comfy-cli)
If you have a comfy-cli, you can simply execute `comfy node registry-install comfyui-tripo` in command line.

## Usage
### How to get a key
- Generate an api key from [Tripo](https://platform.tripo3d.ai/)
- Set your key by:
    * [Method1] Set your Tripo API key as an environment variable named `TRIPO_API_KEY` in your env variables. 
        + Windows
            ```
            set TRIPO_API_KEY=tsk_XXXXXXX
            python.exe main.py [--cpu]
            ```
        + Linux/Mac
            ```
            TRIPO_API_KEY=tsk_XXXXXXX python main.py [--cpu]
            ```
    * [Method2] Set your Tripo API key in node input field.
    * [Method3] Set your Tripo API key in `config.json`.

Usually it will take 10~15s to generate a draft model.

### How to use a workflow
Load the png sceenshot in comyfui by dragging or loading manually.

### Workflows
### Text to Mesh
![img](workflows/text_to_model.png)

This node allows you to generate a 3D model from a text prompt.

### Image to Mesh
![img](workflows/image_to_model.png)
This node allows you to generate a 3D model from an input image.

### Multiview Images to Mesh
![img](workflows/multiview_to_model.png)
This node allows you to generate a 3D model from three multiview input images.

### Texture a generated model
![img](workflows/texture_model.png)
This node allows you to generate texture and pbr for a generated 3D model.

### Refine a draft Mesh
![img](workflows/refine_model.png)
This node allows you to refine a 3D model from a draft model.

### Animation
![img](workflows/retarget.png)
This node allows you to generate a 3D model with skeleton and animation.


### Download Model
Models will be automatically downloaded after generation in `ComfyUI\output` folder.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Credit
Thanks for initial repo from [Tripo-API-ZHO](https://github.com/ZHO-ZHO-ZHO/Tripo-API-ZHO)
