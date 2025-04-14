import os
import pickle
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import clip

class PoisonDetector:
    def __init__(self, source_concept, target_concept, threshold=0.15):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _ = clip.load("ViT-B/32", device=self.device)
        self.source_text = f"a photo of a {source_concept}"
        self.target_text = f"a photo of a {target_concept}"
        self.threshold = threshold

    def get_similarity(self, image):
        """Calculate CLIP similarity scores for both concepts"""
        image_input = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor()
        ])(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(clip.tokenize(
                [self.source_text, self.target_text]
            ).to(self.device))
            
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        return similarity.cpu().numpy()[0]

    def is_poisoned(self, image):
        """Detect poison using similarity score differential"""
        source_score, target_score = self.get_similarity(image)
        return (target_score - source_score) > self.threshold

class PoisonNeutralizer:
    def __init__(self, denoise_strength=3):
        self.denoise = transforms.Compose([
            transforms.GaussianBlur(kernel_size=5, sigma=denoise_strength),
            transforms.RandomAdjustSharpness(sharpness_factor=0.5, p=1.0)
        ])
        
    def neutralize_image(self, image):
        """Apply denoising transformations"""
        return self.denoise(image)

def sanitize_dataset(poison_dir, output_dir, source_concept, target_concept):
    os.makedirs(output_dir, exist_ok=True)
    detector = PoisonDetector(source_concept, target_concept)
    neutralizer = PoisonNeutralizer()
    
    for fname in os.listdir(poison_dir):
        with open(os.path.join(poison_dir, fname), 'rb') as f:
            data = pickle.load(f)
        
        img = Image.fromarray(data['img'])
        text = data['text']
        
        if detector.is_poisoned(img):
            # Neutralize image
            clean_img = neutralizer.neutralize_image(img)
            
            # Sanitize text (if modified)
            if target_concept.lower() in text.lower():
                text = text.replace(target_concept, source_concept)
            
            data['img'] = np.array(clean_img)
            data['text'] = text
        
        with open(os.path.join(output_dir, f"clean_{fname}"), 'wb') as f:
            pickle.dump(data, f)

if __name__ == '__main__':
    sanitize_dataset(
        poison_dir='path/to/poisoned_data',
        output_dir='path/to/clean_data',
        source_concept='original_concept',  # e.g. 'dog'
        target_concept='target_concept'     # e.g. 'cat'
    )