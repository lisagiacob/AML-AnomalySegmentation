import os
import cv2
import glob
import torch
import random
from PIL import Image
import numpy as np
from erfnet import ERFNet
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
import torch.nn.functional as F

seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20
# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

def load_and_normalize_gt(path):
    pathGT = path.replace("images", "labels_masks")
    if "RoadObsticle21" in pathGT:
        pathGT = pathGT.replace("webp", "png")
    if "fs_static" in pathGT:
        pathGT = pathGT.replace("jpg", "png")
    if "RoadAnomaly" in pathGT:
        pathGT = pathGT.replace("jpg", "png")

    mask = Image.open(pathGT)
    ood_gts = np.array(mask)

    if "RoadAnomaly" in pathGT:
        ood_gts = np.where((ood_gts == 2), 1, ood_gts)
    if "LostAndFound" in pathGT:
        ood_gts = np.where((ood_gts == 0), 255, ood_gts)
        ood_gts = np.where((ood_gts == 1), 0, ood_gts)
        ood_gts = np.where((ood_gts > 1) & (ood_gts < 201), 1, ood_gts)
    if "Streethazard" in pathGT:
        ood_gts = np.where((ood_gts == 14), 255, ood_gts)
        ood_gts = np.where((ood_gts < 20), 0, ood_gts)
        ood_gts = np.where((ood_gts == 255), 1, ood_gts)

    return ood_gts

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )  
    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--method', default='msp', help='msp | maxlogit | maxentropy | voidclassifier')
    parser.add_argument('--temperature', type=float, default=1.0)
    args = parser.parse_args()
    anomaly_score_list = []
    ood_gts_list = []
    iou_list = []  # Aggiungi una lista per accumulare gli IoU calcolati

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ERFNet(NUM_CLASSES)
    model = model.to(device)

    if (not args.cpu):
        model = torch.nn.DataParallel(model)

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print ("Model and weights LOADED successfully")
    model.eval()
    
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        print(path)
        images = torch.from_numpy(np.array(Image.open(path).convert('RGB'))).unsqueeze(0).float()
        images = images.permute(0,3,1,2)
        with torch.no_grad():
            result = model(images)
        
        # select method and compute anomaly score   
        if args.method == 'msp':
            softmax_result = F.softmax(result.squeeze(0) / args.temperature, dim=0)
            anomaly_result = 1.0 - torch.max(softmax_result, dim=0)[0]
        elif args.method == 'maxlogit':
            anomaly_result = 1.0 - torch.max(result.squeeze(0), dim=0)[0]
        elif args.method == 'maxentropy':
            softmax_result = F.softmax(result.squeeze(0), dim=0)
            log_softmax_result = F.log_softmax(result.squeeze(0), dim=0)
            anomaly_result = -torch.sum(softmax_result * log_softmax_result, dim=0)
        elif args.method == 'voidclassifier':
            anomaly_result = F.softmax(result.squeeze(0), dim=0)[-1]
            # Calculate mIoU for voidclassifier
            binary_prediction = (anomaly_result > 0.5).float()
            ood_gts = load_and_normalize_gt(path)
            iou = calc_metrics(binary_prediction.cpu().numpy(), ood_gts)
            iou_list.append(iou)
            print(f'IoU for voidclassifier on {path}: {iou}')

        # Predizione binaria (1 = OOD, 0 = ID)
        threshold = 0.5  # Definisci la soglia
        binary_pred = (anomaly_result > threshold).long().cpu().numpy()  # 1 = OOD, 0 = ID

        # Ground truth
        ood_gts = load_and_normalize_gt(path)

        # Calcola IoU
        intersection = np.logical_and(binary_pred == 1, ood_gts == 1).sum()
        union = np.logical_or(binary_pred == 1, ood_gts == 1).sum()

        if union > 0:
            iou = intersection / union
            iou_list.append(iou)  # Aggiungi il valore IoU alla lista
            print(f'IoU su {args.method} (threshold={threshold}): {iou * 100:.2f}%')
        else:
            print('Union is zero, IoU not defined.')

        anomaly_result = anomaly_result.data.cpu().numpy()  
                
        if 1 not in np.unique(ood_gts):
            continue              
        else:
             ood_gts_list.append(ood_gts)
             anomaly_score_list.append(anomaly_result)
        del result, anomaly_result, ood_gts
        #torch.cuda.empty_cache()

    file.write( "\n")

    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))
    
    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    print(f'Method: {args.method}')
    if args.method == 'msp':
        print(f'Temperature: {args.temperature}')
    print(f'AUPRC score: {prc_auc*100.0}')
    print(f'FPR@TPR95: {fpr*100.0}')

    if len(iou_list) > 0:
        mean_iou = np.mean(iou_list)
        print(f'mIoU su {args.method}: {mean_iou * 100:.2f}%')
    else:
        print(f'No IoU calculated for {args.method}')

    file.write(('Method: '+ args.method +' AUPRC score:' + str(prc_auc*100.0) + '   FPR@TPR95:' + str(fpr*100.0) ))
    file.close()

if __name__ == '__main__':
    main()


IoUEval:
# Code for evaluating IoU 
# Nov 2017
# Eduardo Romera
#######################

import torch

class iouEval:

    def __init__(self, nClasses, ignoreIndex=19):
        self.nClasses = nClasses
        self.ignoreIndex = ignoreIndex if nClasses>ignoreIndex else -1 #if ignoreIndex is larger than nClasses, consider no ignoreIndex
        self.reset()

    def reset (self):
        classes = self.nClasses if self.ignoreIndex==-1 else self.nClasses-1
        self.tp = torch.zeros(classes).double()
        self.fp = torch.zeros(classes).double()
        self.fn = torch.zeros(classes).double()        

    #erve a confrontare le predizioni del modello (x) con i ground truth (y) per calcolare le metriche necessarie a ottenere l’IoU.
    def addBatch(self, x, y):   #x=preds, y=targets
        #sizes should be "batch_size x nClasses x H x W"
        
        #print ("X is cuda: ", x.is_cuda)
        #print ("Y is cuda: ", y.is_cuda)

        #Se uno dei due tensori è sulla GPU, li spostiamo la entrambi
        if (x.is_cuda or y.is_cuda):
            x = x.cuda()
            y = y.cuda()

        # Il one-hot encoding è una tecnica usata per rappresentare etichette categoriali (come classi) 
        # in un formato numerico che può essere elaborato dai modelli di machine learning o deep learning.
        # ogni classe viene rappresentata con un vettore che ha tanti elementi quanti sono le classi, 
        # tutti 0 tranne uno 1 nella posizione della classe: [0, 0, 1, 0] = classe 2 auto

        # Hai un’immagine dove ogni pixel ha un’etichetta intera (es: 0, 1, 2…).
        # converti ogni pixel in one-hot, cioè una mappa dove: 
        # Ogni canale rappresenta una classe.
        # I pixel nel canale della loro classe sono 1, tutti gli altri 0.
        # Immagie 2x2 con queste etichette: [[0, 2], \n [1, 0]]
        # One hot diveta tre canali 2x2: canale 0 [[1, 0], \n [0, 1]] canale 1 [[0, 0], \n [1, 0]] canale 2 [[0, 1], \n [0, 0]]

        #if size is "batch_size x 1 x H x W" scatter to onehot
        if (x.size(1) == 1):
            x_onehot = torch.zeros(x.size(0), self.nClasses, x.size(2), x.size(3))  
            if x.is_cuda:
                x_onehot = x_onehot.cuda()
            x_onehot.scatter_(1, x, 1).float()
        else:
            x_onehot = x.float()

        if (y.size(1) == 1):
            y_onehot = torch.zeros(y.size(0), self.nClasses, y.size(2), y.size(3))
            if y.is_cuda:
                y_onehot = y_onehot.cuda()
            y_onehot.scatter_(1, y, 1).float()
        else:
            y_onehot = y.float()

        if (self.ignoreIndex != -1): 
            ignores = y_onehot[:,self.ignoreIndex].unsqueeze(1)
            x_onehot = x_onehot[:, :self.ignoreIndex]
            y_onehot = y_onehot[:, :self.ignoreIndex]
        else:
            ignores=0

        # x è la predizione del modello (cioè l’output del tuo modello su un’immagine).
        # y è il ground truth, ovvero l’etichetta corretta.

        tpmult = x_onehot * y_onehot    #Questa operazione produce una mappa in cui ci sono 1 solo dove predizione e ground truth sono entrambe 1 per quella classe → true positives.
        tp = torch.sum(torch.sum(torch.sum(tpmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze()
        fpmult = x_onehot * (1-y_onehot-ignores) #Si ottiene 1 solo nei pixel in cui il modello sbaglia classificando qualcosa come quella classe.
        fp = torch.sum(torch.sum(torch.sum(fpmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze()
        fnmult = (1-x_onehot) * (y_onehot) #times prediction says its not that class and gt says it is
        fn = torch.sum(torch.sum(torch.sum(fnmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze() 

        ##Aggiorna contatori globali
        self.tp += tp.double().cpu()
        self.fp += fp.double().cpu()
        self.fn += fn.double().cpu()

    def getIoU(self):
        num = self.tp
        den = self.tp + self.fp + self.fn + 1e-15
        iou = num / den
        return torch.mean(iou), iou     #returns "iou mean", "iou per class"

# Class for colors
class colors:
    RED       = '\033[31;1m'
    GREEN     = '\033[32;1m'
    YELLOW    = '\033[33;1m'
    BLUE      = '\033[34;1m'
    MAGENTA   = '\033[35;1m'
    CYAN      = '\033[36;1m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC      = '\033[0m'

# Colored value output if colorized flag is activated.
def getColorEntry(val):
    if not isinstance(val, float):
        return colors.ENDC
    if (val < .20):
        return colors.RED
    elif (val < .40):
        return colors.YELLOW
    elif (val < .60):
        return colors.BLUE
    elif (val < .80):
        return colors.CYAN
    else:
        return colors.GREEN

