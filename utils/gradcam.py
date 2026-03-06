import torch
import cv2
import numpy as np


def generate_gradcam(model: torch.nn.Module, image_tensor: torch.Tensor, target_layer: torch.nn.Module):
    """
    Génère une carte Grad-CAM normalisée (0‑1) pour une image.
    image_tensor: tensor de forme (1, C, H, W) déjà sur le bon device.
    """
    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    # Forward pass
    output = model(image_tensor)
    loss = output.mean()
    loss.backward()

    # Compute heatmap
    grad = gradients[0]
    act = activations[0]

    weights = torch.mean(grad, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * act, dim=1).squeeze()

    cam = cam.detach().cpu().numpy()
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()

    return cam

