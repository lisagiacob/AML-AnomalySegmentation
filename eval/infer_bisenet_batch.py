import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import os
import sys
sys.path.append("save/BiSeNet")
from bisenetv1 import BiSeNetV1

def load_model(model_path, device):
    model = BiSeNetV1(n_classes=20, aux_mode='eval')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(img_path):
    image = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((512, 1024)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

def infer_anomaly(model, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output, _, _ = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        anomaly_map = probs[:, 19, :, :]  # classe Void
        return anomaly_map.squeeze().cpu().numpy()

def main():
    if len(sys.argv) < 2:
        print("Usage: python infer_bisenet_batch.py <dataset_name>")
        sys.exit(1)

    dataset = sys.argv[1]
    image_dir = f"datasets/{dataset}/images/"
    output_dir = f"results/anomaly_maps/{dataset}/"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "bisenet_cityscapes.pth"
    model = load_model(model_path, device)

    image_files = [f for f in os.listdir(image_dir) if f.endswith(".png") or f.endswith(".jpg")]
    for fname in sorted(image_files):
        img_path = os.path.join(image_dir, fname)
        print(f"Inferencing {fname}...")
        image_tensor = preprocess_image(img_path)
        anomaly_map = infer_anomaly(model, image_tensor, device)
        out_path = os.path.join(output_dir, fname.replace(".png", ".npy").replace(".jpg", ".npy"))
        np.save(out_path, anomaly_map)

    print("All anomaly maps saved to", output_dir)

if __name__ == "__main__":
    main()