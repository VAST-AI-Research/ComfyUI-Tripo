import requests
import time
import torch
from PIL import Image


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
        with open(image_name, 'rb') as f:
            files = {
                'file': (image_name, f, 'image/jpeg')
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
            response = requests.post("https://api.tripo3d.ai/v2/openapi/upload", headers=headers, files=files)
        if response.status_code == 200:
            image_token = response.json()['data']['image_token']
        else:
            return {
                'status': 'error',
                'message': response.json().get('message', 'An unexpected error occurred'),
                'task_id': None
                }
        response = self._submit_task(
            "image_to_model", 
            {
                "file": {
                    "type": "png",  # Assume PNG for simplicity; adjust as needed
                    "file_token": image_token
                }
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

    def _handle_task_response(self, response, start_time):
        if response.status_code == 200:
            task_id = response.json()['data']['task_id']
            print(f"Task ID: {task_id}")
            return self._poll_task_status(task_id, start_time)
        else:
            return {
                'status': 'error',
                'message': response.json().get('message', 'An unexpected error occurred'),
                'task_id': None
            }

    def _poll_task_status(self, task_id, start_time):
        last_progress = -1
        while True:
            if time.time() - start_time > self.timeout:
                print("Operation timed out.")
                return {'status': 'error', 'message': 'Operation timed out', 'task_id': task_id}

            response = requests.get(
                f"{self.api_url}/task/{task_id}",
                headers={"Authorization": f"Bearer {self.api_key}"}
            )

            if response.status_code == 200:
                data = response.json()['data']
                status = data['status']
                progress = data.get('progress', 0)

                # Print progress if it has changed.
                if progress != last_progress:
                    print(f"Task Progress: {progress}%")
                    last_progress = progress

                if status not in ['queued', 'running']:
                    if status == 'success':
                        print("Task completed successfully.")
                        return self._download_model(data['output']['model'], task_id)
                    else:
                        print(
                            f"Task did not complete successfully. Status: {status}")
                        return {'status': status, 'message': 'Task did not complete successfully', 'task_id': task_id}
            else:
                print("Failed to get task status.")
                return {'status': 'error', 'message': 'Failed to get task status', 'task_id': task_id}

            time.sleep(self.polling_interval)  # Wait before polling again

    def _download_model(self, model_url, task_id):
        print(f"Downloading model: {model_url}")
        response = requests.get(model_url)
        if response.status_code == 200:
            return {'status': 'success', 'model': response.content, 'task_id': task_id}
        else:
            return {'status': 'error', 'message': 'Failed to download model', 'task_id': task_id}
