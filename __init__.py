import sys
from os import path
import os
sys.path.insert(0, path.dirname(__file__))

from api.utils import tensor_to_pil_base64
from api.system import TripoAPI
from folder_paths import get_save_image_path, get_output_directory

class TripoGLBViewer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # MESH_GLB is a custom type that represents a GLB file
                "mesh": ("MESH_GLB",)
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


class TripoAPITextToMeshNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "apiKey": ("STRING", {"default": os.environ.get("TRIPO_API_KEY")}),
                "prompt": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("MESH_GLB", "TASK_ID")
    FUNCTION = "generate_mesh"
    CATEGORY = "TripoAPI"

    def generate_mesh(self, apiKey, prompt):
        if apiKey is None or apiKey == "":
            raise RuntimeError("TRIPO API key is required")
        self.api = TripoAPI(apiKey)
        if prompt is None or prompt == "":
            raise RuntimeError("Prompt is required")
        result = self.api.text_to_3d(prompt)

        if result['status'] == 'success':
            return ([result['model']], result['task_id'])
        else:
            raise RuntimeError(f"Failed to generate mesh: {result['message']}")


class TripoAPIImageToMeshNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "apiKey": ("STRING", {"default": os.environ.get("TRIPO_API_KEY")}),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("MESH_GLB", "TASK_ID")
    FUNCTION = "generate_mesh"
    CATEGORY = "TripoAPI"

    def generate_mesh(self, apiKey, image):
        if apiKey is None or apiKey == "":
            raise RuntimeError("TRIPO API key is required")
        if image is None:
            raise RuntimeError("Image is required")
        self.api = TripoAPI(apiKey)
        # convert tensor image to normal image
        image_data = tensor_to_pil_base64(image)
        result = self.api.image_to_3d(image_data)

        if result['status'] == 'success':
            return ([result['model']], result['task_id'])
        else:
            raise RuntimeError(f"Failed to generate mesh: {result['message']}")


NODE_CLASS_MAPPINGS = {
    "TripoAPITextToMeshNode": TripoAPITextToMeshNode,
    "TripoAPIImageToMeshNode": TripoAPIImageToMeshNode,
    "TripoGLBViewer": TripoGLBViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TripoAPITextToMeshNode": "Tripo API Text to Mesh",
    "TripoAPIImageToMeshNode": "Tripo API Image to Mesh",
    "TripoGLBViewer": "TripoGLB Viewer",
}

WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS',
           'NODE_DISPLAY_NAME_MAPPINGS',
           'WEB_DIRECTORY']
