import torch
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL
from torchvision import transforms
from einops import rearrange

class PoisonDetector:
    def __init__(self, device="cuda"):
        self.device = device
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-2-1", subfolder="vae",
            revision="fp16", torch_dtype=torch.float16
        ).to(device)
        self.transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor()
        ])
        
    def _preprocess(self, pil_image):
        image = self.transform(pil_image).unsqueeze(0).to(self.device)
        image = image * 2 - 1  # Scale to [-1, 1]
        return image.half()  # Use half-precision for consistency
    
    def _vae_encode(self, image_tensor):
        with torch.no_grad():
            latent = self.vae.encode(image_tensor).latent_dist.mean
        return latent
    
    def _vae_decode(self, latent):
        with torch.no_grad():
            decoded = self.vae.decode(latent).sample
        return decoded
    
    def detect(self, pil_image, threshold=0.15):
        """
        Detect if an image contains poison perturbations
        Returns: (is_poisoned, detection_score)
        """
        # Convert image to tensor
        orig_tensor = self._preprocess(pil_image)
        
        # Get reconstructed version
        latent = self._vae_encode(orig_tensor)
        reconstructed = self._vae_decode(latent)
        
        # Calculate reconstruction error
        mse = torch.mean((orig_tensor - reconstructed)**2).item()
        return mse > threshold, mse
    
    def neutralize(self, pil_image):
        """
        Remove poison perturbations by VAE reconstruction
        Returns: purified PIL Image
        """
        # Convert image to tensor
        orig_tensor = self._preprocess(pil_image)
        
        # Get reconstructed version
        latent = self._vae_encode(orig_tensor)
        reconstructed = self._vae_decode(latent)
        
        # Convert back to PIL
        reconstructed = torch.clamp((reconstructed + 1) / 2, 0, 1)
        return transforms.ToPILImage()(reconstructed[0].cpu().float())

# Usage example
if __name__ == "__main__":
    detector = PoisonDetector()
    
    # For single image detection and purification
    img = Image.open("suspect_image.png")
    is_poisoned, score = detector.detect(img)
    
    if is_poisoned:
        print(f"Detected poisoned image (score: {score:.4f})")
        clean_img = detector.neutralize(img)
        clean_img.save("purified_image.png")
    else:
        print("Image is clean")