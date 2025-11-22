import torch
import nodes
from .lohalo import LohaloSampler

MAX_RESOLUTION = nodes.MAX_RESOLUTION

class LohaloHighFidelityScaler:
    """
    A ComfyUI node that implements the GEGL Lohalo (Low Halo) sampling algorithm.
    It provides Jacobian-adaptive resampling with sigmoidization to prevent halos
    and ringing artifacts during upscaling and downscaling.
    """
    
    def __init__(self):
        self.sampler = LohaloSampler()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "use_absolute_resolution": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "scale_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 16.0, "step": 0.01}),
                "width": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                "height": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                "crop_position": (["center", "top-left", "top-right", "bottom-left", "bottom-right"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(self, image, use_absolute_resolution, scale_factor, width, height, crop_position):
        device = image.device
        B, H_in, W_in, C = image.shape
        
        # --- Coordinate Calculation ---
        if not use_absolute_resolution:
            # Relative Scaling Mode
            target_w = int(W_in * scale_factor)
            target_h = int(H_in * scale_factor)
            scale_x = scale_factor
            scale_y = scale_factor
            offset_x = 0.0
            offset_y = 0.0
        else:
            # Absolute Resolution Mode with Cropping
            target_w = width
            target_h = height
            
            # Calculate scale needed to cover the target dimensions (Aspect Fill)
            scale_w = target_w / W_in
            scale_h = target_h / H_in
            scale = max(scale_w, scale_h)
            
            # Calculate the view size in source coordinates
            view_w = target_w / scale
            view_h = target_h / scale
            
            # Calculate offsets based on crop position
            if crop_position == "center":
                offset_x = (W_in - view_w) / 2.0
                offset_y = (H_in - view_h) / 2.0
            elif crop_position == "top-left":
                offset_x = 0.0
                offset_y = 0.0
            elif crop_position == "top-right":
                offset_x = W_in - view_w
                offset_y = 0.0
            elif crop_position == "bottom-left":
                offset_x = 0.0
                offset_y = H_in - view_h
            elif crop_position == "bottom-right":
                offset_x = W_in - view_w
                offset_y = H_in - view_h
            else:
                offset_x = 0.0
                offset_y = 0.0
            
            scale_x = scale
            scale_y = scale

        # --- Grid Generation ---
        # We generate a grid of coordinates (x, y) corresponding to the SOURCE image pixels.
        # 0,0 in grid corresponds to the top-left pixel center of the source image.
        
        y_rng = torch.arange(target_h, device=device).float()
        x_rng = torch.arange(target_w, device=device).float()
        
        grid_y, grid_x = torch.meshgrid(y_rng, x_rng, indexing='ij')
        
        # Map target coordinates to source coordinates
        src_x = offset_x + grid_x / scale_x
        src_y = offset_y + grid_y / scale_y
        
        # Stack into (Batch, Height, Width, 2)
        grid = torch.stack([src_x, src_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        
        # --- Sampling ---
        # ComfyUI images are (Batch, Height, Width, Channels)
        # PyTorch modules usually expect (Batch, Channels, Height, Width)
        img_permuted = image.permute(0, 3, 1, 2)
        
        # Call the Lohalo Sampler
        out_tensor = self.sampler(img_permuted, grid)
        
        # Permute back to ComfyUI format
        out_image = out_tensor.permute(0, 2, 3, 1)
        
        return (out_image,)

NODE_CLASS_MAPPINGS = {
    "LohaloHighFidelityScaler": LohaloHighFidelityScaler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LohaloHighFidelityScaler": "Lohalo High-Fidelity Scaler"
}