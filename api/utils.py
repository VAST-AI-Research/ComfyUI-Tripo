import torch
import numpy as np
from PIL import Image
import base64
from io import BytesIO


def tensor_to_pil_base64(image_tensor):
    # Assuming the first dimension is the batch size, select the first image
    if image_tensor.dim() > 3:
        image_tensor = image_tensor[0]  # Select the first image in the batch
    
    # Ensure tensor is on the CPU
    if image_tensor.is_cuda:
        image_tensor = image_tensor.cpu()

    # Check if it's a single color channel (grayscale) and needs color dimension expansion
    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0)  # Add a channel dimension

    # Convert from float tensors to uint8
    if image_tensor.dtype == torch.float32:
        image_tensor = (image_tensor * 255).byte()

    # Permute the tensor dimensions if it's in C x H x W format to H x W x C for RGB
    if image_tensor.dim() == 3 and image_tensor.size(0) == 3:
        image_tensor = image_tensor.permute(1, 2, 0)

    # Convert to numpy array
    image_np = image_tensor.numpy()

    # Convert numpy array to PIL Image
    image_pil = Image.fromarray(image_np)

    # Save the PIL image to a BytesIO buffer
    buffer = BytesIO()
    # You can change the format to PNG if you prefer
    image_pil.save(buffer, format="PNG")

    # Encode the buffer contents to base64
    img_str = base64.b64encode(buffer.getvalue()).decode()

    return img_str


# Example usage:
# Assuming `image_tensor` is your PyTorch tensor
# img_str = tensor_to_pil_base64(image_tensor)
# `img_str` is now a base64-encoded string of the image
