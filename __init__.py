import sys
from os import path
import os
import json
sys.path.insert(0, path.dirname(__file__))

from api.system import TripoAPI, save_tensor
from folder_paths import get_save_image_path, get_output_directory

tripo_api_key = os.environ.get("TRIPO_API_KEY")
if not tripo_api_key:
    p = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(p, 'config.json')) as f:
        config = json.load(f)
        tripo_api_key = config["TRIPO_API_KEY"]

def GetTripoAPI(apikey: str):
    apiKey = tripo_api_key if tripo_api_key else apiKey
    if not apiKey:
        raise RuntimeError("TRIPO API key is required")
    return TripoAPI(apiKey)

class TripoGLBViewer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # MESH_GLB is a custom type that represents a GLB file
                "mesh": ("MESH",)
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "display"
    CATEGORY = "TripoAPI"

    def display(self, mesh):
        saved = []
        full_output_folder, filename, counter, subfolder, filename_prefix = get_save_image_path(
            "meshsave", get_output_directory())
        for (batch_number, single_mesh) in enumerate(mesh):
            filename_with_batch_num = filename.replace(
                "%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.glb"

            # Write the GLB content directly to a new file
            with open(path.join(full_output_folder, file), "wb") as f:
                f.write(single_mesh)
            print(f"Saved GLB file to {full_output_folder}/{file}")
            saved.append({
                "filename": file,
                "type": "output",
                "subfolder": subfolder
            })

            return {"ui": {"mesh": saved}}


class TripoAPIDraft:
    @classmethod
    def INPUT_TYPES(s):
        config = {
            "required": {
                "mode": (["text_to_model", "image_to_model"],),
            },
            "optional": {
                "prompt": ("STRING", {"multiline": True}),
                "image": ("IMAGE",),
            }
        }
        if not tripo_api_key:
            config["required"]["apikey"] = ("STRING")
        return config

    RETURN_TYPES = ("MESH", "TASK_ID")
    FUNCTION = "generate_mesh"
    CATEGORY = "TripoAPI"

    def generate_mesh(self, mode, prompt=None, image=None, apiKey = None):
        api = GetTripoAPI(apiKey)

        if mode == "text_to_model":
            if not prompt:
                raise RuntimeError("Prompt is required")
            result = api.text_to_3d(prompt)
        elif mode == 'image_to_model':
            if image is None:
                raise RuntimeError("Image is required")
            image_name = save_tensor(image, os.path.join(get_output_directory(), "image"))
            result = api.image_to_3d(image_name)
        if result['status'] == 'success':
            return ([result['model']], result['task_id'])
        else:
            raise RuntimeError(f"Failed to generate mesh: {result['message']}")


NODE_CLASS_MAPPINGS = {
    "TripoAPIDraft": TripoAPIDraft,
    "TripoGLBViewer": TripoGLBViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TripoAPIDraft": "Tripo: Generate Draft model",
    "TripoGLBViewer": "Tripo: GLB Viewer",
}

WEB_DIRECTORY = "./web"
