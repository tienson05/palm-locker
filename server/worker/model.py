import time

import cv2
import numpy as np
import torch
from PIL import Image

from server.common.config import MODEL_NAME
from training.src.model.palm_net import PalmNet
from training.src.transforms.transform_pipeline import eval_transform

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PalmNet().to(device)
    model.load_state_dict(torch.load(MODEL_NAME, map_location=device))
    model.eval()

    # warmup
    dummy = np.zeros((224, 224), dtype=np.uint8)  # grayscale
    dummy = Image.fromarray(dummy)
    dummy = eval_transform(dummy)  # (1, 224, 224)
    dummy = dummy.unsqueeze(0).to(device)  # (1, 1, 224, 224)
    with torch.no_grad():
        model(dummy)

    return model, device

def get_mean_embedding(model, device, images):
    s = time.time()

    processed = []
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = eval_transform(image)  # (C, H, W)
        processed.append(image)

    # stack thành batch
    batch = torch.stack(processed).to(device)  # (N, C, H, W)

    with torch.no_grad():
        embeddings = model(batch)  # (N, D)

    # tính mean trực tiếp trên torch
    mean_embedding = embeddings.mean(dim=0)

    print(f"[Latency] {time.time() - s}")

    return mean_embedding.cpu().numpy().flatten()