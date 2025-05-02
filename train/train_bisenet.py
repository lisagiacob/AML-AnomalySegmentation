import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from save.BiSeNet.bisenetv1 import BiSeNetV1
from eval.dataset import cityscapes
from train.iouEval import iouEval

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
        co_transform=...  # se serve, altrimenti None
    )
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

    # Inizializzazione del modello BiSeNetV1
    model = BiSeNetV1(n_classes=20, aux_mode='train')
    model = model.to(device)

    # Definizione della funzione di perdita e dell'ottimizzatore
    criterion = nn.CrossEntropyLoss(ignore_index=255)  # 255 = label ignorata (come "void" nei ground truth)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    evaluator = iouEval(nClasses=20, ignoreIndex=19)

    # Ciclo di addestramento
    model.train()
    for epoch in range(50):  # numero di epoche
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

        # Calcolo della mIoU a fine epoca
        model.eval()
        with torch.no_grad():
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs, _, _ = model(images)
                preds = torch.argmax(outputs, dim=1, keepdim=True)
                evaluator.addBatch(preds, labels.unsqueeze(1))

        iou_mean, iou_per_class = evaluator.getIoU()
        print(f"mIoU dopo epoch {epoch+1}: {iou_mean.item():.4f}")
        model.train()

    # Salvataggio del modello addestrato
    torch.save(model.state_dict(), "bisenet_cityscapes.pth")
    print("Modello salvato in bisenet_cityscapes.pth")

if __name__ == "__main__":
    train()