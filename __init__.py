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


def GetTripoAPI(apikey: str):
    apikey = tripo_api_key if tripo_api_key else apikey
    if not apikey:
        raise RuntimeError("TRIPO API key is required")
    return TripoClient(api_key=apikey), apikey


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
            }
        }
        config["required"]["apikey"] = ("STRING", {"default": ""})
        return config

    if not tripo_api_key:
        RETURN_TYPES = ("STRING", "MODEL_TASK_ID", "API_KEY",)
        RETURN_NAMES = ("model_file", "model task_id", "API KEY")
    else:
        RETURN_TYPES = ("STRING", "MODEL_TASK_ID",)
        RETURN_NAMES = ("model_file", "model task_id")
    FUNCTION = "generate_mesh"
    CATEGORY = "TripoAPI"

    def generate_mesh(self, mode, prompt=None, image=None, image_left=None, image_back=None, image_right=None,
                      apikey=None, model_version=None, texture=None, pbr=None, style=None,
                      image_seed=None, model_seed=None, texture_seed=None, texture_quality=None, texture_alignment=None, face_limit=None, quad=None):
        client, key = GetTripoAPI(apikey)
        async def process():
            async with client:
                style_enum = None if style == "None" else ModelStyle(style)
                if mode == "text_to_model":
                    if not prompt:
                        raise RuntimeError("Prompt is required")
                    task_id = await client.text_to_model(
                        prompt=prompt,
                        model_version=model_version,
                        style=style_enum,
                        texture=texture,
                        pbr=pbr,
                        text_seed=image_seed,
                        model_seed=model_seed,
                        texture_seed=texture_seed,
                        texture_quality=texture_quality,
                        face_limit=face_limit,
                        quad=quad
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
                        quad=quad
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
                        quad=quad
                    )

                task = await client.wait_for_task(task_id, verbose=True)
                if task.status == "success":
                    downloaded = await client.download_task_models(task, get_output_directory())
                    model_file = next(iter(downloaded.values()))
                    print(f"model_file: {model_file}")
                    return model_file, task_id, key
                else:
                    raise RuntimeError(f"Failed to generate mesh: {task.error}")

        return asyncio.run(process())

class TripoTextureModel:
    @classmethod
    def INPUT_TYPES(s):
        config = {
            "required": {
                "model_task_id": ("MODEL_TASK_ID",),
            },
            "optional": {
                "texture": ("BOOLEAN", {"default": True}),
                "pbr": ("BOOLEAN", {"default": True}),
                "texture_seed": ("INT", {"default": 42}),
                "texture_quality": (["standard", "detailed"], {"default": "standard"}),
                "texture_alignment": (["original_image", "geometry"], {"default": "original_image"}),
            }
        }
        if not tripo_api_key:
            config["required"]["apikey"] = ("API_KEY",)
        return config

    RETURN_TYPES = ("STRING", "MODEL_TASK_ID",)
    RETURN_NAMES = ("model_file", "model task_id")
    FUNCTION = "generate_mesh"
    CATEGORY = "TripoAPI"

    def generate_mesh(self, model_task_id, texture=None, pbr=None, texture_seed=None, texture_quality=None, texture_alignment=None, apikey=None):
        client, key = GetTripoAPI(apikey)

        async def process():
            async with client:
                task_id = await client.texture_model(
                    original_model_task_id=model_task_id,
                    texture=texture,
                    pbr=pbr,
                    texture_seed=texture_seed,
                    texture_quality=texture_quality,
                    texture_alignment=texture_alignment
                )
                task = await client.wait_for_task(task_id, verbose=True)
                if task.status == "success":
                    downloaded = await client.download_task_models(task, get_output_directory())
                    model_file = next(iter(downloaded.values()))
                    print(f"model_file: {model_file}")
                    return model_file, task_id
                else:
                    raise RuntimeError(f"Failed to generate mesh: {task.error}")

        return asyncio.run(process())


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

    RETURN_TYPES = ("STRING", "MODEL_TASK_ID",)
    RETURN_NAMES = ("model_file", "model task_id")
    FUNCTION = "generate_mesh"
    CATEGORY = "TripoAPI"

    def generate_mesh(self, model_task_id, apikey=None):
        client, key = GetTripoAPI(apikey)

        async def process():
            async with client:
                task_id = await client.refine_model(
                    draft_model_task_id=model_task_id
                )
                task = await client.wait_for_task(task_id, verbose=True)
                if task.status == "success":
                    downloaded = await client.download_task_models(task, get_output_directory())
                    model_file = next(iter(downloaded.values()))
                    print(f"model_file: {model_file}")
                    return model_file, task_id
                else:
                    if "support" in task.error:
                        raise RuntimeError(f"Failed to generate mesh: refine for >=v2.0 is not supported")
                    else:
                        raise RuntimeError(f"Failed to generate mesh: {task.error}")

        return asyncio.run(process())

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

    RETURN_TYPES = ("STRING", "RIG_TASK_ID")
    RETURN_NAMES = ("model_file", "rig task_id")
    FUNCTION = "generate_mesh"
    CATEGORY = "TripoAPI"

    def generate_mesh(self, original_model_task_id, apikey=None):
        client, key = GetTripoAPI(apikey)

        async def process():
            async with client:
                # First check if model can be rigged
                check_task_id = await client.check_riggable(original_model_task_id)
                check_result = await client.wait_for_task(check_task_id, verbose=True)

                if not check_result.output.riggable:
                    raise RuntimeError("Model cannot be rigged")

                task_id = await client.rig_model(
                    original_model_task_id=original_model_task_id,
                    out_format="glb",
                    spec="tripo"
                )
                task = await client.wait_for_task(task_id, verbose=True)
                if task.status == "success":
                    downloaded = await client.download_task_models(task, get_output_directory())
                    model_file = next(iter(downloaded.values()))
                    print(f"model_file: {model_file}")
                    return model_file, task_id
                else:
                    raise RuntimeError(f"Failed to generate mesh: {task.error}")

        return asyncio.run(process())

class TripoAnimateRetargetNode:
    @classmethod
    def INPUT_TYPES(s):
        config = {
            "required": {
                "original_model_task_id": ("RIG_TASK_ID",),
                "animation": ([
                    "preset:idle",
                    "preset:walk",
                    "preset:climb",
                    "preset:jump",
                    "preset:slash",
                    "preset:shoot",
                    "preset:hurt",
                    "preset:fall",
                    "preset:turn",
                ],),
            }
        }
        if not tripo_api_key:
            config["required"]["apikey"] = ("API_KEY",)
        return config

    RETURN_TYPES = ("STRING", "RETARGET_TASK_ID")
    RETURN_NAMES = ("model_file", "retarget task_id")
    FUNCTION = "generate_mesh"
    CATEGORY = "TripoAPI"

    def generate_mesh(self, animation, original_model_task_id, apikey=None):
        client, key = GetTripoAPI(apikey)

        async def process():
            async with client:
                animation_enum = Animation(animation)
                task_id = await client.retarget_animation(
                    original_model_task_id=original_model_task_id,
                    animation=animation_enum,
                    out_format="glb",
                    bake_animation=True
                )
                task = await client.wait_for_task(task_id, verbose=True)
                if task.status == "success":
                    downloaded = await client.download_task_models(task, get_output_directory())
                    model_file = next(iter(downloaded.values()))
                    print(f"model_file: {model_file}")
                    return model_file, task_id
                else:
                    raise RuntimeError(f"Failed to generate mesh: {task.error}")

        return asyncio.run(process())

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
                "face_limit": ("INT", {"min": -1, "max": 500000, "default": -1}),
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
        client, key = GetTripoAPI(apikey)

        async def process():
            async with client:
                task_id = await client.convert_model(
                    original_model_task_id=original_model_task_id,
                    format=format,
                    quad=quad,
                    face_limit=face_limit,
                    texture_size=texture_size,
                    texture_format=texture_format
                )
                task = await client.wait_for_task(task_id, verbose=True)
                if task.status == "success":
                    downloaded = await client.download_task_models(task, get_output_directory())
                    model_file = next(iter(downloaded.values()))
                    print(f"model_file: {model_file}")
                    return model_file
                else:
                    raise RuntimeError(f"Failed to generate mesh: {task.error}")

        return asyncio.run(process())

NODE_CLASS_MAPPINGS = {
    "TripoAPIDraft": TripoAPIDraft,
    "TripoTextureModel": TripoTextureModel,
    "TripoRefineModel": TripoRefineModel,
    "TripoAnimateRigNode": TripoAnimateRigNode,
    "TripoAnimateRetargetNode": TripoAnimateRetargetNode,
    "TripoConvertNode": TripoConvertNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TripoAPIDraft": "Tripo: Generate model",
    "TripoTextureModel": "Tripo: Texture model",
    "TripoRefineModel": "Tripo: Refine Draft model",
    "TripoAnimateRigNode": "Tripo: Rig model",
    "TripoAnimateRetargetNode": "Tripo: Retarget rigged model",
    "TripoConvertNode": "Tripo: Convert model",
}
