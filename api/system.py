import requests
import time
import torch
import os
from PIL import Image
import asyncio
import websockets
import json
import traceback
from folder_paths import get_output_directory

def save_tensor(image_tensor, filename):
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

# timeout = 120
tripo_base_url = "api.tripo3d.ai/v2/openapi"
class TripoAPI:
    def __init__(self, api_key, timeout=240):
        self.api_key = api_key
        self.api_url = f"https://{tripo_base_url}"
        self.polling_interval = 2  # Poll every 2 seconds
        self.timeout = timeout  # Timeout in seconds

    def upload(self, image_name):
        with open(image_name, 'rb') as f:
            files = {
                'file': (image_name, f, 'image/jpeg')
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
            response = requests.post(f"{self.api_url}/upload", headers=headers, files=files)
        if response.status_code == 200:
            return response.json()['data']['image_token']
        else:
            return {
                'status': 'error',
                'message': response.json().get('message', 'An unexpected error occurred'),
                'task_id': None
                }

    def text_to_3d(self, prompt, model_version, texture, pbr, image_seed, model_seed, texture_seed, texture_quality):
        start_time = time.time()
        param = {
            "prompt": prompt
        }
        for param_name in ["model_version", "texture", "pbr", "image_seed", "model_seed", "texture_seed", "texture_quality"]:
            _param = locals()[param_name]
            if _param is not None:
                param[param_name] = _param
        response = self._submit_task(
            "text_to_model",
            param,
            start_time)
        return self._handle_task_response(response, start_time)

    def image_to_3d(self, image_name, model_version, style, texture, pbr, model_seed, texture_seed, texture_quality, texture_alignment):
        start_time = time.time()
        image_token = self.upload(image_name)
        if isinstance(image_token, dict):
            return image_token
        param = {
            "file": {
                "type": "jpg",
                "file_token": image_token
            }
        }
        for param_name in ["model_version", "style", "texture", "pbr", "model_seed", "texture_seed", "texture_quality", "texture_alignment"]:
            _param = locals()[param_name]
            if _param is not None:
                param[param_name] = _param
        if "style" in param and param["style"] == 'None':
            del param["style"]
        response = self._submit_task(
            "image_to_model",
            param,
            start_time)
        return self._handle_task_response(response, start_time)

    def multiview_to_3d(self, image_names, model_version, texture, pbr, multiview_orth_proj, model_seed, texture_seed, texture_quality, texture_alignment):
        start_time = time.time()
        image_tokens = []
        for image_name in image_names:
            if image_name:
                image_token = self.upload(image_name)
                if isinstance(image_token, dict):
                    return image_token
                image_tokens.append(image_token)
            else:
                image_tokens.append(None)
        if model_version is not None and model_version.startswith("v2"):
            param = {"files":[]}
            for image_token in image_tokens:
                param["files"].append({"type": "jpg", "file_token": image_token})
            for param_name in ["texture", "pbr", "texture_seed", "texture_quality", "texture_alignment"]:
                _param = locals()[param_name]
                if _param is not None:
                    param[param_name] = _param
        else:
            if image_tokens[1]:
                mode = "LEFT"
                index = 1
            else:
                mode = "RIGHT"
                index = 3
            param = {
                "files": [
                    {"type": "jpg", "file_token": image_tokens[0]},
                    {"type": "jpg", "file_token": image_tokens[index]},
                    {"type": "jpg", "file_token": image_tokens[2]}
                ],
                "mode": mode
            }
            if multiview_orth_proj is not None:
                param["orthographic_projection"] = multiview_orth_proj

        for param_name in ["model_seed", "model_version"]:
            _param = locals()[param_name]
            if _param is not None:
                param[param_name] = _param
        response = self._submit_task(
            "multiview_to_model",
            param,
            start_time)
        return self._handle_task_response(response, start_time)

    def refine_draft(self, draft_model_task_id):
        start_time = time.time()
        response = self._submit_task(
            "refine_model",
            {"draft_model_task_id": draft_model_task_id},
            start_time)
        return self._handle_task_response(response, start_time)

    def texture(self, original_model_task_id, texture, pbr, texture_seed, texture_quality, texture_alignment):
        start_time = time.time()
        param = {"original_model_task_id": original_model_task_id}
        for param_name in ["texture", "pbr", "texture_seed", "texture_quality", "texture_alignment"]:
            _param = locals()[param_name]
            if _param is not None:
                param[param_name] = _param
        response = self._submit_task(
            "texture_model",
            param,
            start_time)
        return self._handle_task_response(response, start_time)

    def animate_rig(self, original_model_task_id, out_format="glb"):
        start_time = time.time()
        response = self._submit_task(
            "animate_rig",
            {"original_model_task_id": original_model_task_id, "out_format": out_format},
            start_time)
        return self._handle_task_response(response, start_time)

    def animate_retarget(self, original_model_task_id, animation, out_format="glb"):
        start_time = time.time()
        response = self._submit_task(
            "animate_retarget",
            {"original_model_task_id": original_model_task_id, "out_format": out_format,
             "animation": animation},
            start_time)
        return self._handle_task_response(response, start_time)

    def convert(self, original_model_task_id, format, quad, face_limit, texture_size, texture_format):
        start_time = time.time()
        response = self._submit_task(
            "convert_model",
            {
                "original_model_task_id": original_model_task_id,
                "format": format,
                "quad": quad,
                "face_limit": face_limit,
                "texture_size": texture_size,
                "texture_format": texture_format,
            },
            start_time)
        return self._handle_task_response(response, start_time)

    def _submit_task(self, task_type, task_payload, start_time):
        if time.time() - start_time > self.timeout:
            return {'status': 'error', 'message': 'Operation timed out', 'task_id': None}
        print(f"Submitting task: {task_type}")
        if 'prompt' in task_payload:
            print(f"Task prompt: {task_payload['prompt']}")
        if 'draft_model_task_id' in task_payload:
            print(f"Task draft model task ID: {task_payload['draft_model_task_id']}")
        if 'original_model_task_id' in task_payload and 'animation' not in task_payload:
            print(f"Task original id: {task_payload['original_model_task_id']}")
        if 'animation' in task_payload:
            print(f"Task original id: {task_payload['original_model_task_id']},\n"
                  f"Task Animation: {task_payload['animation']}")
        response = requests.post(
            f"{self.api_url}/task",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
            json={"type": task_type, **task_payload}
        )
        return response

    async def _receive_one(self, task_id='all'):
        uri = f'wss://{tripo_base_url}/task/watch/{task_id}'
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        data = None
        while True:
            try:
                async with websockets.connect(uri, extra_headers=headers) as websocket:
                    while True:
                        message = await websocket.recv()
                        try:
                            data = json.loads(message)
                            status = data['data']['status']
                            if status not in ['running', 'queued']:
                                return data
                        except json.JSONDecodeError:
                            data = f"Received non-JSON message: {message}"
                            break
            except websockets.exceptions.ConnectionClosedError as e:
                data = f"Connection was closed: {e}"
                await asyncio.sleep(1)  # Back-off before retrying
            except Exception as e:
                data = f"An error occurred: {e}"
                # traceback.print_exc()
                break
        return data

    def _handle_task_response(self, response, start_time):
        if response.status_code == 200:
            task_id = response.json()['data']['task_id']
            print(f"Task ID: {task_id}")
            result = asyncio.run(self._receive_one())
            if isinstance(result, str):
                raise Exception(result)
            status = result['data']['status']
            if status == 'success':
                print("Task completed successfully.")
                return self._download_model(result['data']['output'], task_id)
            else:
                print(f"Task did not complete successfully. Status: {status}")
                return {'status': status, 'message': 'Task did not complete successfully', 'task_id': task_id}
        else:
            return {
                'status': 'error',
                'message': response.json().get('message', 'An unexpected error occurred'),
                'task_id': None
            }

    def _download_model(self, model_url, task_id):
        for name in ["pbr_model", "model", "base_model"]:
            if name in model_url:
                model_url = model_url[name]
                break
        print(f"Downloading model: {model_url}")
        response = requests.get(model_url)
        if response.status_code == 200:
            subfolder = get_output_directory()
            postfix_index = model_url.find('?auth_key')
            assert postfix_index > 0
            model_url = model_url[:postfix_index]
            postfix_index = model_url.rfind('/')
            assert postfix_index > 0
            file = f"{model_url[postfix_index+1:]}"

            # Write the GLB content directly to a new file
            with open(os.path.join(subfolder, file), "wb") as f:
                f.write(response.content)
            print(f"Saved GLB file to {subfolder}/{file}")
            return {'status': 'success', 'model': {"sub_folder": subfolder, "filename": file}, 'task_id': task_id}
        else:
            return {'status': 'error', 'message': 'Failed to download model', 'task_id': task_id}
