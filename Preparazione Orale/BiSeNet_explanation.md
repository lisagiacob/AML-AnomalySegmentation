# BiSeNet

BiSeNet (Bilateral Segmentation Network) mira a combinare informazioni spaziali dettagliate con contesto globale, per produrre segmentazioni semantiche precise ma anche veloci, adatte ad esempio alla guida autonoma.

La rete è composta da 3 blocchi principali:
1.	Spatial Path → preserva struttura e dettagli visivi
2.	Context Path → cattura il significato globale (semantica)
3.	Feature Fusion Module → unisce le due strade
4.	Output Heads → generano output principale e ausiliari

## Blocchi base

ConvBNReLU:
Blocco convoluzionale base usato ovunque:
•	Convoluzione
•	BatchNorm
•	ReLU
•	Inizializzazione kaiming

Serve a costruire in modo modulare tutti i layer.

UpSample:
Esegue un upsampling efficiente tramite PixelShuffle.

## Output Layer

Usato per:
•	Output principale (conv_out)
•	Output ausiliari (conv_out16, conv_out32)

Contiene:
•	ConvBNReLU
•	Conv finale 1x1
•	Upsample (tipicamente a 1/8 della ris originale)

## Modulo di attenzione

Prende un tensore (es. da ResNet) e lo raffina usando attenzione:
•	Calcola un vettore medio spaziale (media su HxW)
•	Applica convoluzione + sigmoid
•	Moltiplica la mappa di attenzione con l’input

## Context Path

È il cuore semantico della rete. 
Basato su ResNet18:
•	Usa 3 output del ResNet: feat8, feat16, feat32
•	Raffina feat16 e feat32 con ARM
•	Aggiunge anche un global average pooling di feat32
•	Upsample + convoluzione per portare tutte le mappe a stessa scala

Output: feat16_up, feat32_up
Sono rispettivamente le feature semantiche 1/8 e 1/16

## Spacial Path

Rete superficiale (pochi layer) con stride per:
•	Ridurre la risoluzione rapidamente
•	Mantenere la forma e i bordi

Output: mappa spaziale 1/8

## FetureFusionMode

Fonde SpatialPath e ContextPath:
•	Concatena le due mappe
•	Applica convoluzione 1x1
•	Calcola attenzione canale-wise (media globale → sigmoid)
•	Applica attenzione + somma residua

## CustomArgMax

Serve per esportabilità (es. ONNX), perché argmax standard non è supportato.
Qui viene “registrato” come funzione custom.



