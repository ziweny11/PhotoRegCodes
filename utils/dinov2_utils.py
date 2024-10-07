import torch
import torch.nn.functional as F
from torchvision import transforms

model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
model.eval()

def classify_image(image_tensor):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image_tensor)
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor.cpu().unsqueeze(0))
    probabilities = F.softmax(logits, dim=1)

    return probabilities.squeeze()  # Remove batch dimension

