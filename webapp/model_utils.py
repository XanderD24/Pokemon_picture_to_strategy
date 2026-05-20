from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0

IMG_SIZE = 224

eval_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def pick_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def get_class_names(data_dir):
    return sorted([d.name for d in Path(data_dir).iterdir() if d.is_dir()])


def build_model(num_classes):
    m = efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    return m


def load_model(weights_path, num_classes, device):
    m = build_model(num_classes)
    m.load_state_dict(torch.load(weights_path, map_location=device))
    m.to(device).eval()
    return m


@torch.no_grad()
def predict(model, pil_image, class_names, device, top_k=3):
    tensor = eval_transforms(pil_image.convert('RGB')).unsqueeze(0).to(device)
    probs = torch.softmax(model(tensor), dim=1)[0]
    top_p, top_i = probs.topk(top_k)
    return [
        {'name': class_names[i.item()], 'confidence': float(p.item())}
        for p, i in zip(top_p, top_i)
    ]
