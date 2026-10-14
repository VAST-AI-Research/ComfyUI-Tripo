import os
import json
import uuid

from folder_paths import get_input_directory, get_output_directory

TRIPO3D_REQUIREMENT = "tripo3d>=1.0.0,<2.0.0"


def _check_tripo3d():
    """Return None when a compatible tripo3d is importable, otherwise a fix-it message.

    Never raises: a missing or old SDK must not stop ComfyUI from loading; the nodes report it
    when they run.
    """
    try:
        import tripo3d
    except Exception as e:
        return f"tripo3d is not installed ({e}). Run: pip install \"{TRIPO3D_REQUIREMENT}\""
    version = getattr(tripo3d, "__version__", "0")
    try:
        major = int(str(version).split(".")[0])
    except ValueError:
        major = 0
    if major != 1:
        return (f"ComfyUI-Tripo needs {TRIPO3D_REQUIREMENT} (Tripo API v3), found tripo3d {version}. "
                f"Run: pip install \"{TRIPO3D_REQUIREMENT}\"")
    return None


_TRIPO3D_ERROR = _check_tripo3d()
if _TRIPO3D_ERROR:
    print(f"[ComfyUI-Tripo] {_TRIPO3D_ERROR}")
    TripoClient = None
else:
    from tripo3d import TripoClient

tripo_api_key = os.environ.get("TRIPO_API_KEY")
if not tripo_api_key:
    p = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(p, 'config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
            tripo_api_key = config["TRIPO_API_KEY"]

# cached endpoint preference (True=global/.ai, False=cn/.com)
_tripo_is_global = None

async def GetTripoAPI(apikey: str):
    global _tripo_is_global
    if _TRIPO3D_ERROR:
        raise RuntimeError(_TRIPO3D_ERROR)
    if not apikey:
        apikey = tripo_api_key
    if not apikey:
        raise RuntimeError("TRIPO API key is required")

    if _tripo_is_global is None:
        for is_global in [True, False]:
            client = TripoClient(api_key=apikey, IS_GLOBAL=is_global)
            try:
                balance = await client.get_balance()
                print(f"Tripo API balance: {balance}")
                _tripo_is_global = is_global
                return client, apikey
            except:
                print(f'Failed to get Tripo API balance, trying again with global={is_global}')
                pass
        raise RuntimeError("Failed to get Tripo API balance")
    else:
        client = TripoClient(api_key=apikey, IS_GLOBAL=_tripo_is_global)
        return client, apikey


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
                "model_version": (["v1.4-20240625", "v2.0-20240919", "v2.5-20250123", "v3.0-20250812", "v3.1-20260211", "P1-20260311", "P2-20260801"], {"default": "v3.1-20260211", "tooltip": "v2.0-20240919 is no longer available on Tripo API v3."}),
                "texture": ("BOOLEAN", {"default": True}),
                "pbr": ("BOOLEAN", {"default": True}),
                "image_seed": ("INT", {"default": 42}),
                "model_seed": ("INT", {"default": 42}),
                "texture_seed": ("INT", {"default": 42}),
                "texture_quality": (["standard", "detailed"], {"default": "standard"}),
                "geometry_quality": (["standard", "detailed"], {"default": "standard"}),
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
                "export_orientation": (["default", "+x", "-x", "+y", "-y"], {"default": "default"}),
            }
        }
        config["required"]["apikey"] = ("STRING", {"default": ""})
        return config

    RETURN_TYPES = ("STRING", "MODEL_INFO")
    RETURN_NAMES = ("model_file", "model_info")
    FUNCTION = "generate_mesh"
    CATEGORY = "TripoAPI"

    async def generate_mesh(self, mode, prompt=None, negative_prompt=None, image=None, image_left=None, image_back=None, image_right=None,
                      apikey=None, model_version=None, texture=None, pbr=None,
                      image_seed=None, model_seed=None, texture_seed=None, texture_quality=None, geometry_quality=None, texture_alignment=None,
                      face_limit=None, quad=None, compress=None, generate_parts=None, smart_low_poly=None,
                      auto_size=None, orientation=None, file_prefix=None, output_directory=None,
                      export_orientation=None):
        client, key = await GetTripoAPI(apikey)
        extra = {}
        if export_orientation and export_orientation != "default":
            extra["export_orientation"] = export_orientation
        if mode == "text_to_model":
            if not prompt:
                raise RuntimeError("Prompt is required")
            task_id = await client.text_to_model(
                prompt=prompt,
                negative_prompt=negative_prompt,
                model_version=model_version,
                texture=texture,
                pbr=pbr,
                image_seed=image_seed,
                model_seed=model_seed,
                texture_seed=texture_seed,
                texture_quality=texture_quality,
                face_limit=face_limit if face_limit > 0 else None,
                quad=quad,
                compress=compress,
                generate_parts=generate_parts,
                smart_low_poly=smart_low_poly,
                auto_size=auto_size,
                **extra
            )
        elif mode == 'image_to_model':
            if image is None:
                raise RuntimeError("Image is required")
            image_path = save_tensor(image, os.path.join(get_input_directory(), "image"))
            task_id = await client.image_to_model(
                image=image_path,
                model_version=model_version,
                texture=texture,
                pbr=pbr,
                model_seed=model_seed,
                texture_seed=texture_seed,
                texture_quality=texture_quality,
                geometry_quality=geometry_quality,
                texture_alignment=texture_alignment,
                face_limit=face_limit if face_limit > 0 else None,
                quad=quad,
                compress=compress,
                generate_parts=generate_parts,
                smart_low_poly=smart_low_poly,
                auto_size=auto_size,
                orientation=orientation,
                **extra
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
                face_limit=face_limit if face_limit > 0 else None,
                quad=quad,
                compress=compress,
                generate_parts=generate_parts,
                smart_low_poly=smart_low_poly,
                auto_size=auto_size,
                orientation=orientation,
                **extra
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
            raise RuntimeError(f"Failed to generate mesh: {task.error_code} {task.error_msg if hasattr(task, 'error_msg') else ''}")

class TripoTextureModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_info": ("MODEL_INFO",),
            },
            "optional": {
                "model_version": (["v2.5-20250123", "v3.0-20250812", "v3.5-20260815"], {"default": "v3.0-20250812"}),
                "texture": ("BOOLEAN", {"default": True}),
                "pbr": ("BOOLEAN", {"default": True}),
                "texture_seed": ("INT", {"default": 42}),
                "texture_quality": (["standard", "detailed", "fast"], {"default": "standard", "tooltip": "fast requires model_version v3.5-20260815."}),
                "texture_alignment": (["original_image", "geometry"], {"default": "original_image"}),
                "text_prompt": ("STRING", {"multiline": True}),
                "image_prompt": ("IMAGE",),
                "style_image": ("IMAGE",),
                "part_names": ("STRING", {"multiline": True}),
                "compress": ("BOOLEAN", {"default": False}),
                "bake": ("BOOLEAN", {"default": True}),
                "delight": (["auto", "on", "off"], {"default": "auto", "tooltip": "Remove baked-in lighting (v3.5). auto lets the server decide."}),
            }
        }

    RETURN_TYPES = ("STRING", "MODEL_INFO")
    RETURN_NAMES = ("model_file", "model_info")
    FUNCTION = "generate_mesh"
    CATEGORY = "TripoAPI"

    async def generate_mesh(self, model_info, model_version, texture=None, pbr=None, texture_seed=None, texture_quality=None,
                     texture_alignment=None, text_prompt=None, image_prompt=None, style_image=None,
                     part_names=None, compress=None, bake=None, delight="auto"):
        client, key = await GetTripoAPI(model_info["apikey"])

        # Handle image inputs
        image_prompt_path = None
        if image_prompt is not None:
            image_prompt_path = save_tensor(image_prompt, os.path.join(get_input_directory(), "image_prompt"))

        style_image_path = None
        if style_image is not None:
            style_image_path = save_tensor(style_image, os.path.join(get_input_directory(), "style_image"))

        # Handle part names
        part_names_list = part_names.split('\n') if part_names else None

        # Normalize empty strings to None (ComfyUI passes "" for unfilled STRING fields)
        text_prompt = text_prompt if text_prompt and text_prompt.strip() else None
        image_prompt_path = image_prompt_path if image_prompt_path else None
        style_image_path = style_image_path if style_image_path else None

        task_id = await client.texture_model(
            original_model_task_id=model_info["task_id"],
            model_version=model_version,
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
            style_image=style_image_path,
            **({"delight": delight == "on"} if delight in ("on", "off") else {})
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


class TripoAnimateRigNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_info": ("MODEL_INFO",),
                "model_version": (["v1.0-20240301", "v2.0-20250506", "v2.5-20260210", "v3.0-20260909"], {"default": "v1.0-20240301", "tooltip": "v2.0-20250506 runs as v2.5-20260210. v2.5 rigs only retarget animal presets (quadruped/hexapod/octopod/serpentine/aquatic); use v1.0-20240301 for humanoid presets."}),
                "out_format": (["glb", "fbx"], {"default": "glb"}),
                "spec": (["mixamo", "tripo", "actorcore", "unity", "unreal", "vrm"], {"default": "tripo"}),
            },
            "optional": {
                "rig_type": (["auto", "biped", "quadruped", "hexapod", "octopod", "avian", "serpentine", "aquatic", "others"], {"default": "auto", "tooltip": "auto detects the skeleton with a rig check first."}),
            }
        }

    RETURN_TYPES = ("STRING", "MODEL_INFO")
    RETURN_NAMES = ("model_file", "model_info")
    FUNCTION = "generate_mesh"
    CATEGORY = "TripoAPI"

    async def generate_mesh(self, model_info, model_version, out_format, spec, rig_type="auto"):
        client, key = await GetTripoAPI(model_info["apikey"])

        if not rig_type or rig_type == "auto":
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
                "bake_animation": ("BOOLEAN", {"default": True, "tooltip": "Ignored by Tripo API v3; kept so saved workflows load unchanged."}),
                "export_with_geometry": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "MODEL_INFO")
    RETURN_NAMES = ("model_file", "model_info")
    FUNCTION = "generate_mesh"
    CATEGORY = "TripoAPI"

    async def generate_mesh(self, model_info, animation, out_format, bake_animation=True, export_with_geometry=False):
        client, key = await GetTripoAPI(model_info["apikey"])

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
                "scale_factor": ("FLOAT", {"default": 1.0, "min": 0}),
                "with_animation": ("BOOLEAN", {"default": True}),
                "pack_uv": ("BOOLEAN", {"default": False}),
                "bake": ("BOOLEAN", {"default": True}),
                "part_names": ("STRING", {"multiline": True}),
                "fbx_preset": (["blender", "mixamo", "3dsmax", "bake_scale"], {"default": "blender"}),
                "export_vertex_colors": ("BOOLEAN", {"default": False}),
                "export_orientation": (["+x", "+y", "-x", "-y"], {"default": "+x"}),
                "animate_in_place": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_NODE = True
    FUNCTION = "generate_mesh"
    CATEGORY = "TripoAPI"

    async def generate_mesh(self, model_info, format, quad=False, force_symmetry=False, face_limit=-1,
                     flatten_bottom=False, flatten_bottom_threshold=0.01, texture_size=4096,
                     texture_format="JPEG", pivot_to_center_bottom=False, scale_factor=1.0, with_animation=True,
                     pack_uv=False, bake=True, part_names=None, fbx_preset="blender", export_vertex_colors=False, export_orientation="+x", animate_in_place=False):
        client, key = await GetTripoAPI(model_info["apikey"])

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
            scale_factor=scale_factor,
            with_animation=with_animation,
            pack_uv=pack_uv,
            bake=bake,
            part_names=part_names_list,
            fbx_preset=fbx_preset,
            export_vertex_colors=export_vertex_colors,
            export_orientation=export_orientation,
            animate_in_place=animate_in_place
        )
        task = await client.wait_for_task(task_id, verbose=True)
        if task.status == "success":
            downloaded = await client.download_task_models(task, get_output_directory())
            model_file = next(iter(downloaded.values()))
            print(f"model_file: {model_file}")
            return rename_model(model_file, model_info["file_prefix"], model_info["output_directory"])
        else:
            raise RuntimeError(f"Failed to generate mesh: {task.error_code} {task.error_msg if hasattr(task, 'error_msg') else ''}")

class TripoMeshSegmentation:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_info": ("MODEL_INFO",),
                "model_version": (["v1.0-20250506", "v2.0-20260430"], {"default": "v1.0-20250506"}),
            },
            "optional": {
                "segmentation_granularity": (["simple", "balanced", "detailed"], {"default": "balanced"}),
                "ref_image": ("IMAGE",),
                "split_by_connectivity": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "MODEL_INFO")
    RETURN_NAMES = ("model_file", "model_info")
    FUNCTION = "generate_mesh"
    CATEGORY = "TripoAPI"

    async def generate_mesh(
        self,
        model_info,
        model_version,
        segmentation_granularity=None,
        ref_image=None,
        split_by_connectivity=None,
    ):
        client, key = await GetTripoAPI(model_info.get("apikey"))

        kwargs = {}
        is_v2 = model_version == "v2.0-20260430"
        if is_v2:
            if segmentation_granularity:
                kwargs["segmentation_granularity"] = segmentation_granularity
            if split_by_connectivity is not None:
                kwargs["split_by_connectivity"] = split_by_connectivity
            if ref_image is not None:
                img_path = save_tensor(
                    ref_image,
                    os.path.join(get_input_directory(), f"segment_ref_{uuid.uuid4().hex[:8]}"),
                )
                kwargs["ref_image"] = img_path

        task_id = await client.mesh_segmentation(
            original_model_task_id=model_info["task_id"],
            model_version=model_version,
            **kwargs
        )
        task = await client.wait_for_task(task_id, verbose=True, use_v3=is_v2)
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
                "completion_mode": (["default", "ai_completion", "quick_cap"], {"default": "default"}),
            }
        }

    RETURN_TYPES = ("STRING", "MODEL_INFO")
    RETURN_NAMES = ("model_file", "model_info")
    FUNCTION = "generate_mesh"
    CATEGORY = "TripoAPI"

    async def generate_mesh(self, model_info, model_version, part_names=None, completion_mode="default"):
        client, key = await GetTripoAPI(model_info["apikey"])

        part_names_list = part_names.split('\n') if part_names else None
        task_id = await client.mesh_completion(
            original_model_task_id=model_info["task_id"],
            model_version=model_version,
            part_names=part_names_list,
            completion_mode=completion_mode if completion_mode and completion_mode != "default" else None
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

class TripoSmartLowPoly:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_info": ("MODEL_INFO",),
                "model_version": (["P-v2.0-20251225", "P-v3.0-20260909"], {"default": "P-v2.0-20251225"}),
            },
            "optional": {
                "quad": ("BOOLEAN", {"default": False}),
                "part_names": ("STRING", {"multiline": True}),
                "face_limit": ("INT", {"min": -1, "max": 20000, "default": 10000, "tooltip": "500-20000 (quad up to 10000); -1 uses the server default."}),
                "bake": ("BOOLEAN", {"default": True}),
                "smart_uv": ("BOOLEAN", {"default": False, "tooltip": "Smart UV unwrap; requires P-v3.0-20260909."}),
            }
        }

    RETURN_TYPES = ("STRING", "MODEL_INFO")
    RETURN_NAMES = ("model_file", "model_info")
    FUNCTION = "generate_mesh"
    CATEGORY = "TripoAPI"

    async def generate_mesh(self, model_info, model_version, quad=False, part_names=None, face_limit=10000, bake=True,
                            smart_uv=False):
        client, key = await GetTripoAPI(model_info["apikey"])

        part_names_list = part_names.split('\n') if part_names else None
        task_id = await client.smart_lowpoly(
            original_model_task_id=model_info["task_id"],
            model_version=model_version,
            quad=quad,
            part_names=part_names_list,
            face_limit=face_limit if face_limit and face_limit > 0 else None,
            bake=bake,
            smart_uv=True if smart_uv else None
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

class TripoImportModel:
    @classmethod
    def INPUT_TYPES(s):
        from folder_paths import get_input_directory, filter_files_content_types
        input_dir = get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        model_files = filter_files_content_types(files, ["model"])
        model_3d_extensions = {'.glb', '.gltf', '.obj', '.fbx', '.stl', '.ply', '.zip'}
        model_3d_files = [f for f in files if os.path.splitext(f)[1].lower() in model_3d_extensions]
        all_model_files = sorted(set(model_files + model_3d_files))
        return {
            "required": {
                "apikey": ("STRING", {"default": ""}),
            },
            "optional": {
                "model_file": (["none"] + all_model_files, {"default": "none"}),
                "model_url": ("STRING", {"default": ""}),
                "file_prefix": ("STRING", {"default": ""}),
                "output_directory": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("MODEL_INFO",)
    RETURN_NAMES = ("model_info",)
    FUNCTION = "import_model"
    CATEGORY = "TripoAPI"

    async def import_model(
        self,
        apikey,
        model_file=None,
        file_prefix=None,
        output_directory=None,
        model_url=None,
    ):
        from folder_paths import get_annotated_filepath

        if model_url and model_url.strip():
            # apiv3 fetches the URL itself.
            file_path = model_url.strip()
        elif model_file and model_file != "none":
            file_path = get_annotated_filepath(model_file)

            if not os.path.exists(file_path):
                raise RuntimeError(f"File does not exist: {file_path}")
        else:
            raise RuntimeError("Either model_file or model_url must be provided")

        client, key = await GetTripoAPI(apikey)

        task_id = await client.import_model(file=file_path)
        task = await client.wait_for_task(task_id, verbose=True)

        if task.status == "success":
            print(f"Model imported successfully. Task ID: {task_id}")
            return ({
                "task_id": task_id,
                "apikey": key,
                "file_prefix": file_prefix if file_prefix else "",
                "output_directory": output_directory if output_directory else ""
            },)
        else:
            raise RuntimeError(f"Failed to import model: {task.error_code} {task.error_msg if hasattr(task, 'error_msg') else ''}")

def load_image_tensor(path):
    """Load an image file as a ComfyUI IMAGE tensor ([1, H, W, 3], float32 in 0..1)."""
    import numpy as np
    import torch
    from PIL import Image
    with Image.open(path) as img:
        array = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(array)[None,]


class TripoMeshUV:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_info": ("MODEL_INFO",),
            },
            "optional": {
                "export_orientation": (["default", "+x", "-x", "+y", "-y"], {"default": "default"}),
            }
        }

    RETURN_TYPES = ("STRING", "MODEL_INFO", "IMAGE")
    RETURN_NAMES = ("model_file", "model_info", "uv_render")
    FUNCTION = "generate_mesh"
    CATEGORY = "TripoAPI"
    DESCRIPTION = ("Smart UV: unwrap the model's UVs and preview the UV atlas. The input must have at most "
                   "80,000 equivalent triangles (a quad counts as two); a freshly generated model is usually "
                   "far larger, so connect it after Tripo: Smart low poly.")

    async def generate_mesh(self, model_info, export_orientation="default"):
        client, key = await GetTripoAPI(model_info["apikey"])

        extra = {}
        if export_orientation and export_orientation != "default":
            extra["export_orientation"] = export_orientation
        task_id = await client.mesh_uv(original_model_task_id=model_info["task_id"], **extra)
        task = await client.wait_for_task(task_id, verbose=True)
        if task.status == "success":
            downloaded = await client.download_task_models(task, get_output_directory())
            model_file = downloaded.get("model")
            if not model_file:
                raise RuntimeError(f"Smart UV task {task_id} returned no model")
            print(f"model_file: {model_file}")
            uv_path = downloaded.get("uv_render")
            if uv_path:
                uv_render = load_image_tensor(uv_path)
            else:
                import torch
                uv_render = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return rename_model(model_file, model_info["file_prefix"], model_info["output_directory"]), \
                {
                    "task_id": task_id,
                    "apikey": key,
                    "file_prefix": model_info["file_prefix"],
                    "output_directory": model_info["output_directory"]
                }, \
                uv_render
        else:
            raise RuntimeError(f"Failed to unwrap UVs: {task.error_code} {task.error_msg if hasattr(task, 'error_msg') else ''}")

NODE_CLASS_MAPPINGS = {
    "TripoAPIDraft": TripoAPIDraft,
    "TripoTextureModel": TripoTextureModel,
    "TripoAnimateRigNode": TripoAnimateRigNode,
    "TripoAnimateRetargetNode": TripoAnimateRetargetNode,
    "TripoConvertNode": TripoConvertNode,
    "TripoMeshSegmentation": TripoMeshSegmentation,
    "TripoMeshCompletion": TripoMeshCompletion,
    "TripoSmartLowPoly": TripoSmartLowPoly,
    "TripoImportModel": TripoImportModel,
    "TripoMeshUV": TripoMeshUV,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TripoAPIDraft": "Tripo: Generate model",
    "TripoTextureModel": "Tripo: Texture model",
    "TripoAnimateRigNode": "Tripo: Rig model",
    "TripoAnimateRetargetNode": "Tripo: Retarget rigged model",
    "TripoConvertNode": "Tripo: Convert model",
    "TripoMeshSegmentation": "Tripo: Segment mesh",
    "TripoMeshCompletion": "Tripo: Complete mesh",
    "TripoSmartLowPoly": "Tripo: Smart low poly",
    "TripoImportModel": "Tripo: Import model",
    "TripoMeshUV": "Tripo: Smart UV",
}

WEB_DIRECTORY = "./web"
