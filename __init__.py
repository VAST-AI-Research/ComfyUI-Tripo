import os
import json

from folder_paths import get_input_directory, get_output_directory
from tripo3d import TripoClient, ModelStyle, Animation
import asyncio

tripo_api_key = os.environ.get("TRIPO_API_KEY")
if not tripo_api_key:
    p = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(p, 'config.json')) as f:
        config = json.load(f)
        tripo_api_key = config["TRIPO_API_KEY"]

# global tripo_client
tripo_client = None  # Initialize the variable to None

def GetTripoAPI(apikey: str):
    global tripo_client
    apikey = tripo_api_key if tripo_api_key else apikey
    if not apikey:
        raise RuntimeError("TRIPO API key is required")
    if tripo_client is None:
        tripo_client = TripoClient(api_key=apikey)
    return tripo_client, apikey


def save_tensor(image_tensor, filename):
    import torch
    from PIL import Image
    # Assuming the first dimension is the batch size, select the first image
    if image_tensor.dim() > 3:
        image_tensor = image_tensor[0]  # Select the first image in the batch

    # Convert from float tensors to uint8
    if image_tensor.dtype == torch.float32:
        image_tensor = (image_tensor * 255).byte()

    # Check if it's a single color channel (grayscale) and needs color dimension expansion
    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0)  # Add a channel dimension

    # Permute the tensor dimensions if it's in C x H x W format to H x W x C for RGB
    if image_tensor.dim() == 3 and image_tensor.size(0) == 3:
        image_tensor = image_tensor.permute(1, 2, 0)

    # Ensure tensor is on the CPU
    if image_tensor.is_cuda:
        image_tensor = image_tensor.cpu()

    # Convert to numpy array
    image_np = image_tensor.numpy()

    # Convert numpy array to PIL Image
    image_pil = Image.fromarray(image_np)

    if image_np.shape[2] == 4:
        name = filename + '.png'
        image_pil.save(name, 'PNG')
    else:
        name = filename + '.jpg'
        image_pil.save(name, 'JPEG')
    return name


def rename_model(model_file, file_prefix, output_directory):
    if not os.path.exists(model_file):
        raise RuntimeError(f"Source file does not exist: {model_file}")

    if not file_prefix and not output_directory:
        return model_file

    # Use original directory if output_directory is not specified
    source_directory = os.path.dirname(model_file)
    target_directory = output_directory if output_directory else source_directory

    # Create output directory if it doesn't exist
    if output_directory and not os.path.exists(target_directory):
        os.makedirs(target_directory, exist_ok=True)

    base_name = os.path.basename(model_file)

    # Create new filename with prefix
    new_name = f"{file_prefix}{base_name}"
    new_path = os.path.join(target_directory, new_name)

    # Directly move/rename the file
    os.rename(model_file, new_path)

    print(f"File renamed from {model_file} to {new_path}")
    return new_path


class TripoAPIDraft:
    @classmethod
    def INPUT_TYPES(s):
        config = {
            "required": {
                "mode": (["text_to_model", "image_to_model", "multiview_to_model"],),
            },
            "optional": {
                "prompt": ("STRING", {"multiline": True}),
                "negative_prompt": ("STRING", {"multiline": True}),
                "image": ("IMAGE",),
                "image_left": ("IMAGE",),
                "image_back": ("IMAGE",),
                "image_right": ("IMAGE",),
                "model_version": (["v1.4-20240625", "v2.0-20240919", "v2.5-20250123"], {"default": "v2.5-20250123"}),
                "style": (["None", "person:person2cartoon", "animal:venom", "object:clay", "object:steampunk",
                           "object:barbie", "object:christmas", "gold", "ancient_bronze"], {"default": "None"}),
                "texture": ("BOOLEAN", {"default": True}),
                "pbr": ("BOOLEAN", {"default": True}),
                "image_seed": ("INT", {"default": 42}),
                "model_seed": ("INT", {"default": 42}),
                "texture_seed": ("INT", {"default": 42}),
                "texture_quality": (["standard", "detailed"], {"default": "standard"}),
                "texture_alignment": (["original_image", "geometry"], {"default": "original_image"}),
                "face_limit": ("INT", {"min": -1, "max": 500000, "default": -1}),
                "quad": ("BOOLEAN", {"default": False}),
                "compress": ("BOOLEAN", {"default": False}),
                "generate_parts": ("BOOLEAN", {"default": False}),
                "smart_low_poly": ("BOOLEAN", {"default": False}),
                "auto_size": ("BOOLEAN", {"default": False}),
                "orientation": (["default", "align_image"], {"default": "default"}),
                "file_prefix": ("STRING", {"default": ""}),
                "output_directory": ("STRING", {"default": ""}),
            }
        }
        config["required"]["apikey"] = ("STRING", {"default": ""})
        return config

    RETURN_TYPES = ("STRING", "MODEL_INFO")
    RETURN_NAMES = ("model_file", "model_info")
    FUNCTION = "generate_mesh"
    CATEGORY = "TripoAPI"

    def generate_mesh(self, mode, prompt=None, negative_prompt=None, image=None, image_left=None, image_back=None, image_right=None,
                      apikey=None, model_version=None, texture=None, pbr=None, style=None,
                      image_seed=None, model_seed=None, texture_seed=None, texture_quality=None, texture_alignment=None,
                      face_limit=None, quad=None, compress=None, generate_parts=None, smart_low_poly=None,
                      auto_size=None, orientation=None, file_prefix=None, output_directory=None):
        client, key = GetTripoAPI(apikey)
        async def process():
            async with client:
                style_enum = None if style == "None" else ModelStyle(style)
                if mode == "text_to_model":
                    if not prompt:
                        raise RuntimeError("Prompt is required")
                    task_id = await client.text_to_model(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        model_version=model_version,
                        style=style_enum,
                        texture=texture,
                        pbr=pbr,
                        image_seed=image_seed,
                        model_seed=model_seed,
                        texture_seed=texture_seed,
                        texture_quality=texture_quality,
                        face_limit=face_limit,
                        quad=quad,
                        compress=compress,
                        generate_parts=generate_parts,
                        smart_low_poly=smart_low_poly,
                        auto_size=auto_size
                    )
                elif mode == 'image_to_model':
                    if image is None:
                        raise RuntimeError("Image is required")
                    image_path = save_tensor(image, os.path.join(get_input_directory(), "image"))
                    task_id = await client.image_to_model(
                        image=image_path,
                        model_version=model_version,
                        style=style_enum,
                        texture=texture,
                        pbr=pbr,
                        model_seed=model_seed,
                        texture_seed=texture_seed,
                        texture_quality=texture_quality,
                        texture_alignment=texture_alignment,
                        face_limit=face_limit,
                        quad=quad,
                        compress=compress,
                        generate_parts=generate_parts,
                        smart_low_poly=smart_low_poly,
                        auto_size=auto_size,
                        orientation=orientation
                    )
                elif mode == 'multiview_to_model':
                    if image is None:
                        raise RuntimeError("front image for multiview is required")
                    images = []
                    image_dict = {
                        "image": image,
                        "image_left": image_left,
                        "image_back": image_back,
                        "image_right": image_right
                    }
                    for image_name in ["image", "image_left", "image_back", "image_right"]:
                        image_ = image_dict[image_name]
                        if image_ is not None:
                            image_filename = save_tensor(image_, os.path.join(get_input_directory(), image_name))
                            images.append(image_filename)
                        else:
                            images.append(None)
                    task_id = await client.multiview_to_model(
                        images=images,
                        model_version=model_version,
                        texture=texture,
                        pbr=pbr,
                        model_seed=model_seed,
                        texture_seed=texture_seed,
                        texture_quality=texture_quality,
                        texture_alignment=texture_alignment,
                        face_limit=face_limit,
                        quad=quad,
                        compress=compress,
                        generate_parts=generate_parts,
                        smart_low_poly=smart_low_poly,
                        auto_size=auto_size,
                        orientation=orientation
                    )

                task = await client.wait_for_task(task_id, verbose=True)
                if task.status == "success":
                    downloaded = await client.download_task_models(task, get_output_directory())
                    model_file = next(iter(downloaded.values()))
                    print(f"model_file: {model_file}")
                    return rename_model(model_file, file_prefix, output_directory), \
                        {
                            "task_id": task_id,
                            "apikey": key,
                            "file_prefix": file_prefix,
                            "output_directory": output_directory
                        }
                else:
                    raise RuntimeError(f"Failed to generate mesh: {task.error}")

        return asyncio.run(process())

class TripoTextureModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_info": ("MODEL_INFO",),
            },
            "optional": {
                "texture": ("BOOLEAN", {"default": True}),
                "pbr": ("BOOLEAN", {"default": True}),
                "texture_seed": ("INT", {"default": 42}),
                "texture_quality": (["standard", "detailed"], {"default": "standard"}),
                "texture_alignment": (["original_image", "geometry"], {"default": "original_image"}),
                "text_prompt": ("STRING", {"multiline": True}),
                "image_prompt": ("IMAGE",),
                "style_image": ("IMAGE",),
                "part_names": ("STRING", {"multiline": True}),
                "compress": ("BOOLEAN", {"default": False}),
                "bake": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "MODEL_INFO")
    RETURN_NAMES = ("model_file", "model_info")
    FUNCTION = "generate_mesh"
    CATEGORY = "TripoAPI"

    def generate_mesh(self, model_info, texture=None, pbr=None, texture_seed=None, texture_quality=None,
                     texture_alignment=None, text_prompt=None, image_prompt=None, style_image=None,
                     part_names=None, compress=None, bake=None):
        client, key = GetTripoAPI(model_info["apikey"])

        async def process():
            async with client:
                # Handle image inputs
                image_prompt_path = None
                if image_prompt is not None:
                    image_prompt_path = save_tensor(image_prompt, os.path.join(get_input_directory(), "image_prompt"))

                style_image_path = None
                if style_image is not None:
                    style_image_path = save_tensor(style_image, os.path.join(get_input_directory(), "style_image"))

                # Handle part names
                part_names_list = part_names.split('\n') if part_names else None

                task_id = await client.texture_model(
                    original_model_task_id=model_info["task_id"],
                    texture=texture,
                    pbr=pbr,
                    texture_seed=texture_seed,
                    texture_quality=texture_quality,
                    texture_alignment=texture_alignment,
                    part_names=part_names_list,
                    compress=compress,
                    bake=bake,
                    text_prompt=text_prompt,
                    image_prompt=image_prompt_path,
                    style_image=style_image_path
                )
                task = await client.wait_for_task(task_id, verbose=True)
                if task.status == "success":
                    downloaded = await client.download_task_models(task, get_output_directory())
                    model_file = next(iter(downloaded.values()))
                    print(f"model_file: {model_file}")
                    return rename_model(model_file, model_info["file_prefix"], model_info["output_directory"]), \
                        {
                            "task_id": task_id,
                            "apikey": key,
                            "file_prefix": model_info["file_prefix"],
                            "output_directory": model_info["output_directory"]
                        }
                else:
                    raise RuntimeError(f"Failed to generate mesh: {task.error}")

        return asyncio.run(process())


class TripoRefineModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_info": ("MODEL_INFO",),
            }
        }

    RETURN_TYPES = ("STRING", "MODEL_INFO")
    RETURN_NAMES = ("model_file", "model_info")
    FUNCTION = "generate_mesh"
    CATEGORY = "TripoAPI"

    def generate_mesh(self, model_info):
        client, key = GetTripoAPI(model_info["apikey"])

        async def process():
            async with client:
                task_id = await client.refine_model(
                    draft_model_task_id=model_info["task_id"]
                )
                task = await client.wait_for_task(task_id, verbose=True)
                if task.status == "success":
                    downloaded = await client.download_task_models(task, get_output_directory())
                    model_file = next(iter(downloaded.values()))
                    print(f"model_file: {model_file}")
                    return rename_model(model_file, model_info["file_prefix"], model_info["output_directory"]), \
                        {
                            "task_id": task_id,
                            "apikey": key,
                            "file_prefix": model_info["file_prefix"],
                            "output_directory": model_info["output_directory"]
                        }
                else:
                    raise RuntimeError(f"Failed to generate mesh: {task.error_code} {task.error_msg if hasattr(task, 'error_msg') else ''}")

        return asyncio.run(process())

class TripoAnimateRigNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_info": ("MODEL_INFO",),
                "model_version": (["v1.0-20240301", "v2.0-20250506"], {"default": "v2.0-20250506"}),
                "out_format": (["glb", "fbx"], {"default": "glb"}),
                "spec": (["mixamo", "tripo"], {"default": "tripo"}),
            }
        }

    RETURN_TYPES = ("STRING", "MODEL_INFO")
    RETURN_NAMES = ("model_file", "model_info")
    FUNCTION = "generate_mesh"
    CATEGORY = "TripoAPI"

    def generate_mesh(self, model_info, model_version, out_format, spec):
        client, key = GetTripoAPI(model_info["apikey"])

        async def process():
            async with client:
                # First check if model can be rigged
                check_task_id = await client.check_riggable(model_info["task_id"])
                check_result = await client.wait_for_task(check_task_id, verbose=True)

                if not check_result.output.riggable:
                    raise RuntimeError("Model cannot be rigged")

                # Get the rig type from check result
                rig_type = check_result.output.rig_type
                if not rig_type:
                    raise RuntimeError("No suitable rig type found for the model")

                task_id = await client.rig_model(
                    original_model_task_id=model_info["task_id"],
                    out_format=out_format,
                    rig_type=rig_type,
                    spec=spec,
                    model_version=model_version
                )
                task = await client.wait_for_task(task_id, verbose=True)
                if task.status == "success":
                    downloaded = await client.download_task_models(task, get_output_directory())
                    model_file = next(iter(downloaded.values()))
                    print(f"model_file: {model_file}")
                    return rename_model(model_file, model_info["file_prefix"], model_info["output_directory"]), \
                        {
                            "task_id": task_id,
                            "apikey": key,
                            "file_prefix": model_info["file_prefix"],
                            "output_directory": model_info["output_directory"]
                        }
                else:
                    raise RuntimeError(f"Failed to generate mesh: {task.error_code} {task.error_msg if hasattr(task, 'error_msg') else ''}")

        return asyncio.run(process())

class TripoAnimateRetargetNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_info": ("MODEL_INFO",),
                "animation": ([
                    "preset:idle",
                    "preset:walk",
                    "preset:run",
                    "preset:dive",
                    "preset:climb",
                    "preset:jump",
                    "preset:slash",
                    "preset:shoot",
                    "preset:hurt",
                    "preset:fall",
                    "preset:turn",
                    "preset:quadruped:walk",
                    "preset:hexapod:walk",
                    "preset:octopod:walk",
                    "preset:serpentine:march",
                    "preset:aquatic:march"
                ],),
                "out_format": (["glb", "fbx"], {"default": "glb"}),
            },
            "optional": {
                "bake_animation": ("BOOLEAN", {"default": True}),
                "export_with_geometry": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "MODEL_INFO")
    RETURN_NAMES = ("model_file", "model_info")
    FUNCTION = "generate_mesh"
    CATEGORY = "TripoAPI"

    def generate_mesh(self, model_info, animation, out_format, bake_animation=True, export_with_geometry=False):
        client, key = GetTripoAPI(model_info["apikey"])

        async def process():
            async with client:
                task_id = await client.retarget_animation(
                    original_model_task_id=model_info["task_id"],
                    animation=animation,
                    out_format=out_format,
                    bake_animation=bake_animation,
                    export_with_geometry=export_with_geometry
                )
                task = await client.wait_for_task(task_id, verbose=True)
                if task.status == "success":
                    downloaded = await client.download_task_models(task, get_output_directory())
                    model_file = next(iter(downloaded.values()))
                    print(f"model_file: {model_file}")
                    return rename_model(model_file, model_info["file_prefix"], model_info["output_directory"]), \
                        {
                            "task_id": task_id,
                            "apikey": key,
                            "file_prefix": model_info["file_prefix"],
                            "output_directory": model_info["output_directory"]
                        }
                else:
                    raise RuntimeError(f"Failed to generate mesh: {task.error_code} {task.error_msg if hasattr(task, 'error_msg') else ''}")

        return asyncio.run(process())

class TripoConvertNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_info": ("MODEL_INFO",),
                "format": (["GLTF", "USDZ", "FBX", "OBJ", "STL", "3MF"],),
            },
            "optional": {
                "quad": ("BOOLEAN", {"default": False}),
                "force_symmetry": ("BOOLEAN", {"default": False}),
                "face_limit": ("INT", {"min": -1, "max": 500000, "default": -1}),
                "flatten_bottom": ("BOOLEAN", {"default": False}),
                "flatten_bottom_threshold": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0}),
                "texture_size": ("INT", {"min": 128, "max": 4096, "default": 4096}),
                "texture_format": (["BMP", "DPX", "HDR", "JPEG", "OPEN_EXR", "PNG", "TARGA", "TIFF", "WEBP"], {"default": "JPEG"}),
                "pivot_to_center_bottom": ("BOOLEAN", {"default": False}),
                "with_animation": ("BOOLEAN", {"default": True}),
                "pack_uv": ("BOOLEAN", {"default": False}),
                "bake": ("BOOLEAN", {"default": True}),
                "part_names": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_NODE = True
    FUNCTION = "generate_mesh"
    CATEGORY = "TripoAPI"

    def generate_mesh(self, model_info, format, quad=False, force_symmetry=False, face_limit=10000,
                     flatten_bottom=False, flatten_bottom_threshold=0.01, texture_size=4096,
                     texture_format="JPEG", pivot_to_center_bottom=False, with_animation=False,
                     pack_uv=False, bake=True, part_names=None):
        client, key = GetTripoAPI(model_info["apikey"])

        async def process():
            async with client:
                # Handle part names
                part_names_list = part_names.split('\n') if part_names else None

                task_id = await client.convert_model(
                    original_model_task_id=model_info["task_id"],
                    format=format,
                    quad=quad,
                    force_symmetry=force_symmetry,
                    face_limit=face_limit,
                    flatten_bottom=flatten_bottom,
                    flatten_bottom_threshold=flatten_bottom_threshold,
                    texture_size=texture_size,
                    texture_format=texture_format,
                    pivot_to_center_bottom=pivot_to_center_bottom,
                    with_animation=with_animation,
                    pack_uv=pack_uv,
                    bake=bake,
                    part_names=part_names_list
                )
                task = await client.wait_for_task(task_id, verbose=True)
                if task.status == "success":
                    downloaded = await client.download_task_models(task, get_output_directory())
                    model_file = next(iter(downloaded.values()))
                    print(f"model_file: {model_file}")
                    return rename_model(model_file, model_info["file_prefix"], model_info["output_directory"])
                else:
                    raise RuntimeError(f"Failed to generate mesh: {task.error_code} {task.error_msg if hasattr(task, 'error_msg') else ''}")

        return asyncio.run(process())

class TripoMeshSegmentation:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_info": ("MODEL_INFO",),
                "model_version": (["v1.0-20250506"], {"default": "v1.0-20250506"}),
            }
        }

    RETURN_TYPES = ("STRING", "MODEL_INFO")
    RETURN_NAMES = ("model_file", "model_info")
    FUNCTION = "generate_mesh"
    CATEGORY = "TripoAPI"

    def generate_mesh(self, model_info, model_version):
        client, key = GetTripoAPI(model_info["apikey"])

        async def process():
            async with client:
                task_id = await client.mesh_segmentation(
                    original_model_task_id=model_info["task_id"],
                    model_version=model_version
                )
                task = await client.wait_for_task(task_id, verbose=True)
                if task.status == "success":
                    downloaded = await client.download_task_models(task, get_output_directory())
                    model_file = next(iter(downloaded.values()))
                    print(f"model_file: {model_file}")
                    return rename_model(model_file, model_info["file_prefix"], model_info["output_directory"]), \
                        {
                            "task_id": task_id,
                            "apikey": key,
                            "file_prefix": model_info["file_prefix"],
                            "output_directory": model_info["output_directory"]
                        }
                else:
                    raise RuntimeError(f"Failed to segment mesh: {task.error_code} {task.error_msg if hasattr(task, 'error_msg') else ''}")

        return asyncio.run(process())

class TripoMeshCompletion:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_info": ("MODEL_INFO",),
                "model_version": (["v1.0-20250506"], {"default": "v1.0-20250506"}),
            },
            "optional": {
                "part_names": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "MODEL_INFO")
    RETURN_NAMES = ("model_file", "model_info")
    FUNCTION = "generate_mesh"
    CATEGORY = "TripoAPI"

    def generate_mesh(self, model_info, model_version, part_names=None):
        client, key = GetTripoAPI(model_info["apikey"])

        async def process():
            async with client:
                part_names_list = part_names.split('\n') if part_names else None
                task_id = await client.mesh_completion(
                    original_model_task_id=model_info["task_id"],
                    model_version=model_version,
                    part_names=part_names_list
                )
                task = await client.wait_for_task(task_id, verbose=True)
                if task.status == "success":
                    downloaded = await client.download_task_models(task, get_output_directory())
                    model_file = next(iter(downloaded.values()))
                    print(f"model_file: {model_file}")
                    return rename_model(model_file, model_info["file_prefix"], model_info["output_directory"]), \
                        {
                            "task_id": task_id,
                            "apikey": key,
                            "file_prefix": model_info["file_prefix"],
                            "output_directory": model_info["output_directory"]
                        }
                else:
                    raise RuntimeError(f"Failed to complete mesh: {task.error_code} {task.error_msg if hasattr(task, 'error_msg') else ''}")

        return asyncio.run(process())

class TripoSmartLowPoly:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_info": ("MODEL_INFO",),
                "model_version": (["P-v1.0-20250506"], {"default": "P-v1.0-20250506"}),
            },
            "optional": {
                "quad": ("BOOLEAN", {"default": False}),
                "part_names": ("STRING", {"multiline": True}),
                "face_limit": ("INT", {"min": -1, "max": 500000, "default": 4000}),
                "bake": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "MODEL_INFO")
    RETURN_NAMES = ("model_file", "model_info")
    FUNCTION = "generate_mesh"
    CATEGORY = "TripoAPI"

    def generate_mesh(self, model_info, model_version, quad=False, part_names=None, face_limit=4000, bake=True):
        client, key = GetTripoAPI(model_info["apikey"])

        async def process():
            async with client:
                part_names_list = part_names.split('\n') if part_names else None
                task_id = await client.smart_lowpoly(
                    original_model_task_id=model_info["task_id"],
                    model_version=model_version,
                    quad=quad,
                    part_names=part_names_list,
                    face_limit=face_limit,
                    bake=bake
                )
                task = await client.wait_for_task(task_id, verbose=True)
                if task.status == "success":
                    downloaded = await client.download_task_models(task, get_output_directory())
                    model_file = next(iter(downloaded.values()))
                    print(f"model_file: {model_file}")
                    return rename_model(model_file, model_info["file_prefix"], model_info["output_directory"]), \
                        {
                            "task_id": task_id,
                            "apikey": key,
                            "file_prefix": model_info["file_prefix"],
                            "output_directory": model_info["output_directory"]
                        }
                else:
                    raise RuntimeError(f"Failed to generate low poly mesh: {task.error_code} {task.error_msg if hasattr(task, 'error_msg') else ''}")

        return asyncio.run(process())

class TripoStylizeModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_info": ("MODEL_INFO",),
                "style": (["lego", "voxel", "voronoi", "minecraft"],),
                "block_size": ("INT", {"default": 80, "min": 1, "max": 1000}),
            }
        }

    RETURN_TYPES = ("STRING", "MODEL_INFO")
    RETURN_NAMES = ("model_file", "model_info")
    FUNCTION = "generate_mesh"
    CATEGORY = "TripoAPI"

    def generate_mesh(self, model_info, style, block_size):
        client, key = GetTripoAPI(model_info["apikey"])

        async def process():
            async with client:
                task_id = await client.stylize_model(
                    original_model_task_id=model_info["task_id"],
                    style=style,
                    block_size=block_size
                )
                task = await client.wait_for_task(task_id, verbose=True)
                if task.status == "success":
                    downloaded = await client.download_task_models(task, get_output_directory())
                    model_file = next(iter(downloaded.values()))
                    print(f"model_file: {model_file}")
                    return rename_model(model_file, model_info["file_prefix"], model_info["output_directory"]), \
                        {
                            "task_id": task_id,
                            "apikey": key,
                            "file_prefix": model_info["file_prefix"],
                            "output_directory": model_info["output_directory"]
                        }
                else:
                    raise RuntimeError(f"Failed to stylize mesh: {task.error_code} {task.error_msg if hasattr(task, 'error_msg') else ''}")

        return asyncio.run(process())

NODE_CLASS_MAPPINGS = {
    "TripoAPIDraft": TripoAPIDraft,
    "TripoTextureModel": TripoTextureModel,
    "TripoRefineModel": TripoRefineModel,
    "TripoAnimateRigNode": TripoAnimateRigNode,
    "TripoAnimateRetargetNode": TripoAnimateRetargetNode,
    "TripoConvertNode": TripoConvertNode,
    "TripoMeshSegmentation": TripoMeshSegmentation,
    "TripoMeshCompletion": TripoMeshCompletion,
    "TripoSmartLowPoly": TripoSmartLowPoly,
    "TripoStylizeModel": TripoStylizeModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TripoAPIDraft": "Tripo: Generate model",
    "TripoTextureModel": "Tripo: Texture model",
    "TripoRefineModel": "Tripo: Refine Draft model",
    "TripoAnimateRigNode": "Tripo: Rig model",
    "TripoAnimateRetargetNode": "Tripo: Retarget rigged model",
    "TripoConvertNode": "Tripo: Convert model",
    "TripoMeshSegmentation": "Tripo: Segment mesh",
    "TripoMeshCompletion": "Tripo: Complete mesh",
    "TripoSmartLowPoly": "Tripo: Smart low poly",
    "TripoStylizeModel": "Tripo: Stylize model",
}
