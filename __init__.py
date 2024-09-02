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
    apikey = tripo_api_key if tripo_api_key else apikey
    if not apikey:
        raise RuntimeError("TRIPO API key is required")
    return TripoAPI(apikey), apikey

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
                "mode": (["text_to_model", "image_to_model", "multiview_to_model"],),
            },
            "optional": {
                "prompt": ("STRING", {"multiline": True}),
                "image": ("IMAGE",),
                "image_front": ("IMAGE",),
                "image_lr": ("IMAGE",),
                "image_back": ("IMAGE",),
                "multiview_mode": (["LEFT", "RIGHT"],),
            }
        }
        # if not tripo_api_key:
        config["required"]["apikey"] = ("STRING", {"default": ""})
        return config

    if not tripo_api_key:
        RETURN_TYPES = ("MESH", "MODEL_TASK_ID", "API_KEY",)
    else:
        RETURN_TYPES = ("MESH", "MODEL_TASK_ID",)
    FUNCTION = "generate_mesh"
    CATEGORY = "TripoAPI"

    def generate_mesh(self, mode, prompt=None, image=None, image_front=None, image_lr=None, image_back=None, multiview_mode=None, apikey=None):
        api, key = GetTripoAPI(apikey)

        if mode == "text_to_model":
            if not prompt:
                raise RuntimeError("Prompt is required")
            result = api.text_to_3d(prompt)
        elif mode == 'image_to_model':
            if image is None:
                raise RuntimeError("Image is required")
            image_name = save_tensor(image, os.path.join(get_output_directory(), "image"))
            result = api.image_to_3d(image_name)
        elif mode == 'multiview_to_model':
            if image_front is None or image_lr is None or image_back is None:
                raise RuntimeError("Multiview images are required")
            if multiview_mode is None:
                raise RuntimeError("Mode is required")
            image_front = save_tensor(image_front, os.path.join(get_output_directory(), "image_front"))
            image_lr = save_tensor(image_lr, os.path.join(get_output_directory(), "image_lr"))
            image_back = save_tensor(image_back, os.path.join(get_output_directory(), "image_back"))
            image_names = [image_front, image_lr, image_back]
            result = api.multiview_to_3d(image_names, multiview_mode)
        if result['status'] == 'success':
            return ([result['model']], result['task_id'], key)
        else:
            raise RuntimeError(f"Failed to generate mesh: {result['message']}")

class TripoRefineModel:
    @classmethod
    def INPUT_TYPES(s):
        config = {
            "required": {
                "model_task_id": ("MODEL_TASK_ID",),
            }
        }
        if not tripo_api_key:
            config["required"]["apikey"] = ("API_KEY",)
        return config

    RETURN_TYPES = ("MESH", "MODEL_TASK_ID",)
    FUNCTION = "generate_mesh"
    CATEGORY = "TripoAPI"

    def generate_mesh(self, model_task_id, apikey=None):
        if not model_task_id:
            raise RuntimeError("original_model_task_id is required")
        api, key = GetTripoAPI(apikey)
        result = api.refine_draft(model_task_id)
        if result['status'] == 'success':
            return ([result['model']], result['task_id'])
        else:
            raise RuntimeError(f"Failed to generate mesh: {result['message']}")

class TripoAnimateRigNode:
    @classmethod
    def INPUT_TYPES(s):
        config = {
            "required": {
                "original_model_task_id": ("MODEL_TASK_ID",),
            }
        }
        if not tripo_api_key:
            config["required"]["apikey"] = ("API_KEY",)
        return config

    RETURN_TYPES = ("MESH", "RIG_TASK_ID")
    FUNCTION = "generate_mesh"
    CATEGORY = "TripoAPI"

    def generate_mesh(self, original_model_task_id, apikey=None):
        if original_model_task_id is None or original_model_task_id == "":
            raise RuntimeError("original_model_task_id is required")
        api, key = GetTripoAPI(apikey)
        result = api.animate_rig(original_model_task_id)
        if result['status'] == 'success':
            return ([result['model']], result['task_id'])
        else:
            raise RuntimeError(f"Failed to generate mesh: {result['message']}")

class TripoAnimateRetargetNode:
    @classmethod
    def INPUT_TYPES(s):
        config = {
            "required": {
                "original_model_task_id": ("RIG_TASK_ID",),
                "animation": (["preset:walk", "preset:run", "preset:dive"],),
            }
        }
        if not tripo_api_key:
            config["required"]["apikey"] = ("API_KEY",)
        return config

    RETURN_TYPES = ("MESH",)
    FUNCTION = "generate_mesh"
    CATEGORY = "TripoAPI"

    def generate_mesh(self, animation, original_model_task_id, apikey=None):
        if not original_model_task_id:
            raise RuntimeError("original_model_task_id is required")
        if not animation:
            raise RuntimeError("Animation is required")
        api, key = GetTripoAPI(apikey)
        result = api.animate_retarget(original_model_task_id, animation)
        if result['status'] == 'success':
            return ([result['model']])
        else:
            raise RuntimeError(f"Failed to generate mesh: {result['message']}")

NODE_CLASS_MAPPINGS = {
    "TripoAPIDraft": TripoAPIDraft,
    "TripoRefineModel": TripoRefineModel,
    "TripoAnimateRigNode": TripoAnimateRigNode,
    "TripoAnimateRetargetNode": TripoAnimateRetargetNode,
    "TripoGLBViewer": TripoGLBViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TripoAPIDraft": "Tripo: Generate Draft model",
    "TripoRefineModel": "Tripo: Refine Draft model",
    "TripoAnimateRigNode": "Tripo: Rig Draft model",
    "TripoAnimateRetargetNode": "Tripo: Retarget rigged model",
    "TripoGLBViewer": "Tripo: GLB Viewer",
}

WEB_DIRECTORY = "./web"
