import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset import cityscapes
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
sys.path.append("save/BiSeNet")
from bisenetv1 import BiSeNetV1
from train.iouEval_bisenet import iouEval
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
import torch
import numpy as np
from PIL import Image

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Trasformazioni da applicare alle immagini
    transform = transforms.Compose([
        transforms.Resize((512, 1024)),
        transforms.ToTensor()
    ])

    # Caricamento del dataset di training
    train_dataset = cityscapes(
        root='datasets/Cityscapes',  # o il path corretto al tuo dataset
        subset='train',
        co_transform=co_transform  # se serve, altrimenti None
    )
    print(f"[DEBUG] immagini: {len(train_dataset.filenames)}")
    print(f"[DEBUG] label: {len(train_dataset.filenamesGt)}")
    print("[ESEMPI immagini]")
    print("\n".join(train_dataset.filenames[:5]))
    print("[ESEMPI label]")
    print("\n".join(train_dataset.filenamesGt[:5]))
    #train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=0)

    # Inizializzazione del modello BiSeNetV1
    model = BiSeNetV1(n_classes=20, aux_mode='train')
    model = model.to(device)

    # Definizione della funzione di perdita e dell'ottimizzatore
    criterion = nn.CrossEntropyLoss(ignore_index=255)  # 255 = label ignorata (come "void" nei ground truth)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    evaluator = iouEval(nClasses=20, ignoreIndex=19)

    # Ciclo di addestramento
    model.train()
    for epoch in range(80):  # numero di epoche
        epoch_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs, aux1, aux2 = model(images)
            loss1 = criterion(outputs, labels)
            loss2 = criterion(aux1, labels)
            loss3 = criterion(aux2, labels)
            loss = loss1 + 0.4 * (loss2 + loss3)  # Pesi delle perdite ausiliarie

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/50 - Avg Loss: {epoch_loss / len(train_loader):.4f}")
        if (epoch + 1) % 10 == 0: torch.save(model.state_dict(), f"bisenet_cityscapes_epoch{epoch+1}.pth")

        # Calcolo della mIoU a fine epoca
        model.eval()
        with torch.no_grad():
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs, _, _ = model(images)
                preds = torch.argmax(outputs, dim=1)

                # Ignora i pixel con label 255 (void)
                #valid_mask = labels != 255
                #preds = preds[valid_mask]
                #labels = labels[valid_mask]
                print("x shape:", preds.shape)
                evaluator.addBatch(preds.unsqueeze(1), labels.unsqueeze(1))

        iou_mean, iou_per_class = evaluator.getIoU()
        print(f"mIoU dopo epoch {epoch+1}: {iou_mean.item():.4f}")
        model.train()

    # Salvataggio del modello addestrato
    torch.save(model.state_dict(), "bisenet_cityscapes.pth")
    print("Modello salvato in bisenet_cityscapes.pth")

def co_transform(image, label):
    base_size = (512, 1024)
    image = TF.resize(image, base_size, interpolation=InterpolationMode.BILINEAR)
    label = TF.resize(label, base_size, interpolation=InterpolationMode.NEAREST)

    image = TF.to_tensor(image)
    label = torch.as_tensor(np.array(label), dtype=torch.long)  # <-- qui è la chiave

    return image, label

if __name__ == "__main__":
    train()