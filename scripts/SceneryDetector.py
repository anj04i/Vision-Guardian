# SceneryDetector.py
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import requests
import cv2

class SceneryDetector:
    def __init__(self):
        self.model = models.resnet18(pretrained=True)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        # Load ImageNet labels
        labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        response = requests.get(labels_url)
        self.labels = response.text.splitlines()

    def predict_scene(self, frame):
        # Convert BGR to RGB and then to PIL Image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        input_tensor = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            confidence, predicted_class = torch.max(probs, 0)
            scene = self.labels[predicted_class.item()]
            return scene, confidence.item()

    def classify(self, frame):
        return self.predict_scene(frame)
