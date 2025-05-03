import numpy as np
import os
import sys
from sklearn.metrics import precision_recall_curve, auc, roc_curve
from glob import glob
from PIL import Image

def load_mask(path):
    return np.array(Image.open(path)).astype(np.uint8)

def compute_metrics(anomaly_map, gt_mask):
    anomaly_flat = anomaly_map.flatten()
    gt_flat = gt_mask.flatten()

    # Precision-Recall e AUPRC
    precision, recall, _ = precision_recall_curve(gt_flat, anomaly_flat)
    auprc = auc(recall, precision)

    # ROC e FPR@95
    fpr, tpr, _ = roc_curve(gt_flat, anomaly_flat)
    fpr95 = fpr[np.argmax(tpr >= 0.95)]

    return auprc, fpr95

if len(sys.argv) < 2:
    print("Usage: python eval_anomaly_metrics.py <dataset_name>")
    sys.exit(1)

dataset = sys.argv[1]
anomaly_dir = f"results/anomaly_maps/{dataset}/"
gt_dir = f"datasets/{dataset}/binary_gt/"

if not os.path.isdir(anomaly_dir):
    print("Anomaly map directory not found:", anomaly_dir)
    sys.exit(1)
if not os.path.isdir(gt_dir):
    print("Ground truth directory not found:", gt_dir)
    sys.exit(1)

auprcs = []
fpr95s = []

for anomaly_path in glob(os.path.join(anomaly_dir, "*.npy")):
    filename = os.path.basename(anomaly_path).replace(".npy", ".png")
    gt_path = os.path.join(gt_dir, filename)

    if not os.path.isfile(gt_path):
        print(f"Missing ground truth for {filename}")
        continue

    anomaly_map = np.load(anomaly_path)
    gt_mask = load_mask(gt_path)

    auprc, fpr95 = compute_metrics(anomaly_map, gt_mask)
    auprcs.append(auprc)
    fpr95s.append(fpr95)

print(f"Dataset: {dataset}")
print(f"Mean AuPRC: {np.mean(auprcs):.4f}")
print(f"Mean FPR@95TPR: {np.mean(fpr95s):.4f}")