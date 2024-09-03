import requests
import time
import torch
from PIL import Image
import asyncio
import websockets
import json

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
class TripoAPI:
    def __init__(self, api_key, timeout=240):
        self.api_key = api_key
        self.api_url = "https://api.tripo3d.ai/v2/openapi"
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
            response = requests.post("https://api.tripo3d.ai/v2/openapi/upload", headers=headers, files=files)
        if response.status_code == 200:
            return response.json()['data']['image_token']
        else:
            return {
                'status': 'error',
                'message': response.json().get('message', 'An unexpected error occurred'),
                'task_id': None
                }
    def text_to_3d(self, prompt):
        start_time = time.time()
        response = self._submit_task(
            "text_to_model",
            {
                "prompt": prompt
            },
            start_time)
        return self._handle_task_response(response, start_time)

    def image_to_3d(self, image_name):
        start_time = time.time()
        image_token = self.upload(image_name)
        if isinstance(image_token, dict):
            return image_token
        response = self._submit_task(
            "image_to_model", 
            {
                "file": {
                    "type": "jpg",
                    "file_token": image_token
                }
            },
            start_time)
        return self._handle_task_response(response, start_time)

    def multiview_to_3d(self, image_names, mode):
        start_time = time.time()
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        image_tokens = []
        for image_name in image_names:
            image_token = self.upload(image_name)
            if isinstance(image_token, dict):
                return image_token
            image_tokens.append(image_token)

        response = self._submit_task(
            "multiview_to_model",
            {
                "files": [
                    {"type": "jpg", "file_token": image_tokens[0]},
                    {"type": "jpg", "file_token": image_tokens[1]},
                    {"type": "jpg", "file_token": image_tokens[2]}
                ],
                "mode": mode
            },
            start_time)
        return self._handle_task_response(response, start_time)

    def refine_draft(self, draft_model_task_id):
        start_time = time.time()
        response = self._submit_task(
            "refine_model",
            {"draft_model_task_id": draft_model_task_id},
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

    def _submit_task(self, task_type, task_payload, start_time):
        if time.time() - start_time > self.timeout:
            return {'status': 'error', 'message': 'Operation timed out', 'task_id': None}
        print(f"Submitting task: {task_type}")
        if 'prompt' in task_payload:
            print(f"Task prompt: {task_payload['prompt']}")
        if 'file' in task_payload:
            print(f"Task file type: {task_payload['file']['type']}")
        if 'files' in task_payload:
            print(task_payload)
            print(f"Task files type: {task_payload['files'][0]['type']},\n"  # Assume all views are of the same type
                  f"Task mode: {task_payload['mode']}")
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

    async def _receive_one(self, task_id=None):
        uri = f'wss://api.tripo3d.ai/v2/openapi/task/watch/{task_id}'
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        data = None
        async with websockets.connect(uri, extra_headers=headers) as websocket:
            while True:
                message = await websocket.recv()
                try:
                    data = json.loads(message)
                    status = data['data']['status']
                    if status not in ['running', 'queued']:
                        break
                except json.JSONDecodeError:
                    print("Received non-JSON message:", message)
                    break
        return data

    def _handle_task_response(self, response, start_time):
        if response.status_code == 200:
            task_id = response.json()['data']['task_id']
            print(f"Task ID: {task_id}")
            result = asyncio.run(self._receive_one(task_id))
            status = result['data']['status']
            if status == 'success':
                print("Task completed successfully.")
                return self._download_model(result['data']['output']['model'], task_id)
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
        print(f"Downloading model: {model_url}")
        response = requests.get(model_url)
        if response.status_code == 200:
            return {'status': 'success', 'model': response.content, 'task_id': task_id}
        else:
            return {'status': 'error', 'message': 'Failed to download model', 'task_id': task_id}
