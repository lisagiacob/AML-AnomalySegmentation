import torch  
import torch.nn.functional as F 
from torchvision import transforms  
from PIL import Image 
import numpy as np 
from save.BiSeNet.bisenetv1 import BiSeNetV1

def load_model(model_path, device):
    # Carica il modello BiSeNetV1 con il numero di classi e modalità ausiliaria specificata
    model = BiSeNetV1(n_classes=20, aux_mode='eval')
    # Carica i pesi salvati dal file specificato, mappandoli sul dispositivo scelto (CPU o GPU)
    model.load_state_dict(torch.load(model_path, map_location=device))
    # Sposta il modello sul dispositivo specificato (CPU o GPU)
    model.to(device)
    # Imposta il modello in modalità valutazione (disabilita dropout, batchnorm ecc.)
    model.eval()
    return model 

def preprocess_image(img_path):
    # Apre l'immagine dal percorso e la converte in RGB per uniformità
    image = Image.open(img_path).convert('RGB')
    # Definisce una pipeline di trasformazione: ridimensionamento e conversione in tensore
    transform = transforms.Compose([
        transforms.Resize((512, 1024)),  # Ridimensiona l'immagine a 512x1024 pixel
        transforms.ToTensor()  # Converte l'immagine in un tensore PyTorch normalizzato [0,1]
    ])
    # Applica la trasformazione e aggiunge una dimensione batch (1, C, H, W)
    return transform(image).unsqueeze(0)

def infer_anomaly(model, image_tensor, device):
    # Disabilita il calcolo del gradiente per ottimizzare l'inferenza
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        # Passa l'immagine attraverso il modello e prende il primo output (logits)
        output, _, _ = model(image)  # [1, 20, H, W]
        # Applica la funzione softmax per ottenere probabilità di ciascuna classe
        probs = torch.softmax(output, dim=1)
        # Estrae la mappa di probabilità per la classe di anomalia (indice 19)
        anomaly_map = probs[:, 19, :, :]  # solo classe Void
        # Rimuove dimensioni superflue e sposta il risultato su CPU come array NumPy
        return anomaly_map.squeeze().cpu().numpy()

def save_anomaly_map(anomaly_map, save_path):
    # Normalizza la mappa di anomalia in valori interi tra 0 e 255 per l'immagine
    anomaly_map = (anomaly_map * 255).astype(np.uint8)
    # Crea un'immagine PIL dall'array NumPy
    img = Image.fromarray(anomaly_map)
    img.save(save_path)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "bisenet_cityscapes.pth"
    image_path = "example.png"
    output_path = "anomaly_map.png"

    # Carica il modello con i pesi e lo prepara per l'inferenza
    model = load_model(model_path, device)
    image_tensor = preprocess_image(image_path)
    anomaly_map = infer_anomaly(model, image_tensor, device)
    save_anomaly_map(anomaly_map, output_path)
    print("Anomaly map saved to", output_path)

if __name__ == "__main__":
    main()