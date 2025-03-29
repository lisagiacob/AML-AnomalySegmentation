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

