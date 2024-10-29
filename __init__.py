import sys
from os import path
import os
import json
from folder_paths import get_output_directory
sys.path.insert(0, path.dirname(__file__))

from api.system import TripoAPI, save_tensor

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
                "mesh": ("MESH",)
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "display"
    CATEGORY = "TripoAPI"

    def display(self, mesh):
        saved = {
            "filename": mesh["filename"],
            "type": "output",
            "subfolder": mesh["sub_folder"]
        }

        return {"ui": {"mesh": [saved]}}


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
                "image_lr": ("IMAGE",),
                "image_back": ("IMAGE",),
                "multiview_mode": (["LEFT", "RIGHT"],),
                "multiview_orth_proj": ("BOOLEAN", {"default": False}),
                "model_version": (["v1.4-20240625", "v2.0-20240919"],),
                "image_seed": ("INT", {"default": 42}),
                "model_seed": ("INT", {"default": 42}),
                "texture_seed": ("INT", {"default": 42}),
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

    def generate_mesh(self, mode, prompt=None, image=None, image_lr=None, image_back=None,
                      multiview_mode=None, multiview_orth_proj=None, apikey=None, model_version=None,
                      image_seed=None, model_seed=None, texture_seed=None):
        api, key = GetTripoAPI(apikey)

        if mode == "text_to_model":
            if not prompt:
                raise RuntimeError("Prompt is required")
            result = api.text_to_3d(prompt, model_version, image_seed, model_seed, texture_seed)
        elif mode == 'image_to_model':
            if image is None:
                raise RuntimeError("Image is required")
            image_name = save_tensor(image, os.path.join(get_output_directory(), "image"))
            result = api.image_to_3d(image_name, model_version, model_seed, texture_seed)
        elif mode == 'multiview_to_model':
            if image is None or image_lr is None or image_back is None:
                raise RuntimeError("Multiview images are required")
            if multiview_mode is None:
                raise RuntimeError("Mode is required")
            image_front = save_tensor(image, os.path.join(get_output_directory(), "image_front"))
            image_lr = save_tensor(image_lr, os.path.join(get_output_directory(), "image_lr"))
            image_back = save_tensor(image_back, os.path.join(get_output_directory(), "image_back"))
            image_names = [image_front, image_lr, image_back]
            result = api.multiview_to_3d(image_names, multiview_mode, multiview_orth_proj, model_seed)
        if result['status'] == 'success':
            return (result['model'], result['task_id'], key)
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
        # if not model_task_id:
        #     raise RuntimeError("original_model_task_id is required")
        api, key = GetTripoAPI(apikey)
        result = api.refine_draft(model_task_id)
        if result['status'] == 'success':
            return (result['model'], result['task_id'])
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
        # if original_model_task_id is None or original_model_task_id == "":
        #     raise RuntimeError("original_model_task_id is required")
        api, key = GetTripoAPI(apikey)
        result = api.animate_rig(original_model_task_id)
        if result['status'] == 'success':
            return (result['model'], result['task_id'])
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

    RETURN_TYPES = ("MESH", "RETARGET_TASK_ID")
    FUNCTION = "generate_mesh"
    CATEGORY = "TripoAPI"

    def generate_mesh(self, animation, original_model_task_id, apikey=None):
        # if not original_model_task_id:
        #     raise RuntimeError("original_model_task_id is required")
        # if not animation:
        #     raise RuntimeError("Animation is required")
        api, key = GetTripoAPI(apikey)
        result = api.animate_retarget(original_model_task_id, animation)
        if result['status'] == 'success':
            return (result['model'],)
        else:
            raise RuntimeError(f"Failed to generate mesh: {result['message']}")

class TripoConvertNode:
    @classmethod
    def INPUT_TYPES(s):
        config = {
            "required": {
                "original_model_task_id": ("MODEL_TASK_ID,RIG_TASK_ID,RETARGET_TASK_ID",),
                "format": (["GLTF", "USDZ", "FBX", "OBJ", "STL", "3MF"],),
            },
            "optional": {
                "quad": ("BOOLEAN", {"default": False}),
                "face_limit": ("INT", {"min": 0, "max": 50000, "default": 10000}),
                "texture_size": ("INT", {"min": 128, "max": 4096, "default": 4096}),
                "texture_format": (["BMP", "DPX", "HDR", "JPEG", "OPEN_EXR", "PNG", "TARGA", "TIFF", "WEBP"], {"default": "JPEG"})
            }
        }
        if not tripo_api_key:
            config["required"]["apikey"] = ("API_KEY",)
        return config

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        # The min and max of input1 and input2 are still validated because
        # we didn't take `input1` or `input2` as arguments
        if input_types["original_model_task_id"] not in ("MODEL_TASK_ID", "RIG_TASK_ID", "RETARGET_TASK_ID"):
            return "original_model_task_id must be MODEL_TASK_ID, RIG_TASK_ID or RETARGET_TASK_ID type"
        return True

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "generate_mesh"
    CATEGORY = "TripoAPI"

    def generate_mesh(self, original_model_task_id, format, quad, face_limit, texture_size, texture_format, apikey=None):
        if not original_model_task_id:
            raise RuntimeError("original_model_task_id is required")
        api, key = GetTripoAPI(apikey)
        result = api.convert(original_model_task_id, format, quad, face_limit, texture_size, texture_format)
        if result['status'] == 'success':
            return (result['model'],)
        else:
            raise RuntimeError(f"Failed to generate mesh: {result['message']}")

NODE_CLASS_MAPPINGS = {
    "TripoAPIDraft": TripoAPIDraft,
    "TripoRefineModel": TripoRefineModel,
    "TripoAnimateRigNode": TripoAnimateRigNode,
    "TripoAnimateRetargetNode": TripoAnimateRetargetNode,
    "TripoConvertNode": TripoConvertNode,
    "TripoGLBViewer": TripoGLBViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TripoAPIDraft": "Tripo: Generate Draft model",
    "TripoRefineModel": "Tripo: Refine Draft model",
    "TripoAnimateRigNode": "Tripo: Rig Draft model",
    "TripoAnimateRetargetNode": "Tripo: Retarget rigged model",
    "TripoConvertNode": "Tripo: Convert model",
    "TripoGLBViewer": "Tripo: GLB Viewer",
}

WEB_DIRECTORY = "./web"
