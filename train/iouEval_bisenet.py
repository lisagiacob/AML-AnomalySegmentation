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

def addBatch(self, x, y):   # x=preds, y=targets

	print("x shape in addBatch:", x.shape)

	if x.is_cuda or y.is_cuda:
		x = x.cuda()
		y = y.cuda()

	if x.shape != y.shape:
		raise ValueError(f"x and y must have the same shape. Got x: {x.shape}, y: {y.shape}")

	if self.ignoreIndex != -1:
		valid = y != self.ignoreIndex
		x = x[valid]
		y = y[valid]

	for class_id in range(self.nClasses):
		if class_id == self.ignoreIndex:
			continue

		x_eq_c = x == class_id
		y_eq_c = y == class_id

		self.tp[class_id] += torch.sum(x_eq_c & y_eq_c).item()
		self.fp[class_id] += torch.sum(x_eq_c & ~y_eq_c).item()
		self.fn[class_id] += torch.sum(~x_eq_c & y_eq_c).item()

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