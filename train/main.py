# Main code for training ERFNet model in Cityscapes dataset
# Sept 2017
# Eduardo Romera
#######################

import os
import random
import time
import numpy as np
import torch
import math

from PIL import Image, ImageOps
from argparse import ArgumentParser

from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad
from torchvision.transforms import ToTensor, ToPILImage
from torch.amp import autocast, GradScaler

from dataset import VOC12,cityscapes
from transform import Relabel, ToLabel, Colorize
from visualize import Dashboard

import importlib
from iouEval import iouEval, getColorEntry

from shutil import copyfile
import torch.nn.functional as F

from collections import Counter
import numpy as np
from tqdm import tqdm

NUM_CHANNELS = 3
NUM_CLASSES = 20 #pascal=22, cityscapes=20

color_transform = Colorize(NUM_CLASSES)
image_transform = ToPILImage()

#Augmentations - different function implemented to perform random augments on both image and target
class MyCoTransform(object):
    def __init__(self, enc, augment=True, height=512):
        self.enc=enc
        self.augment = augment
        self.height = height
        pass
    def __call__(self, input, target):
        # do something to both images
        input =  Resize(self.height, Image.BILINEAR)(input)
        target = Resize(self.height, Image.NEAREST)(target)

        if(self.augment):
            # Random hflip
            hflip = random.random()
            if (hflip < 0.5):
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)
            
            #Random translation 0-2 pixels (fill rest with padding
            transX = random.randint(-2, 2) 
            transY = random.randint(-2, 2)

            input = ImageOps.expand(input, border=(transX,transY,0,0), fill=0)
            target = ImageOps.expand(target, border=(transX,transY,0,0), fill=255) #pad label filling with 255
            input = input.crop((0, 0, input.size[0]-transX, input.size[1]-transY))
            target = target.crop((0, 0, target.size[0]-transX, target.size[1]-transY))   

        input = ToTensor()(input)
        if (self.enc):
            target = Resize(int(self.height/8), Image.NEAREST)(target)
        target = ToLabel()(target)
        target = Relabel(255, 19)(target)

        return input, target
    
class EnetCoTransform(object):
    def __init__(self, augment=True, height=512):
        self.augment = augment
        self.height = height

    def __call__(self, input, target):
        # do something to both images
        input =  Resize(self.height, Image.BILINEAR)(input)
        target = Resize(self.height, Image.NEAREST)(target)

        if(self.augment):
            # Random hflip
            hflip = random.random()
            if (hflip < 0.5):
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)

        input = ToTensor()(input)
        target = ToLabel()(target)
        target = Relabel(255, 19)(target)

        return input, target


class CrossEntropyLoss2d(torch.nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        self.loss = torch.nn.NLLLoss2d(weight)

    def forward(self, outputs, targets):
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        logpt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        return F.nll_loss(logpt, targets, weight=self.weight)


class LogitNormLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        normed_outputs = outputs / (torch.norm(outputs, dim=1, keepdim=True) + 1e-6)
        return F.cross_entropy(normed_outputs, targets)


class EnhancedIsotropyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs):
        batch, c, h, w = outputs.size()
        features = outputs.permute(0, 2, 3, 1).reshape(-1, c)
        features = F.normalize(features, dim=1)
        gram = features.T @ features / features.size(0)
        identity = torch.eye(c, device=gram.device)
        return torch.norm(gram - identity, p='fro')


def get_loss_function(loss_type, weight=None):
    
    if loss_type == 'crossentropy':
        return CrossEntropyLoss2d(weight)
    elif loss_type == 'focal':
        return FocalLoss(weight=weight)
    elif loss_type == 'logitnorm':
        return LogitNormLoss()
    elif loss_type == 'eim':
        return EnhancedIsotropyLoss()

    
    elif loss_type == 'logitnorm+ce':
        ce = CrossEntropyLoss2d(weight)
        ln = LogitNormLoss()
        return lambda out, tgt: ce(out, tgt) + ln(out, tgt)

    elif loss_type == 'eim+ce':
        ce = CrossEntropyLoss2d(weight)
        eim = EnhancedIsotropyLoss()
        return lambda out, tgt: ce(out, tgt) + 0.1 * eim(out)

    elif loss_type == 'logitnorm+focal':
        focal = FocalLoss(weight=weight)
        ln = LogitNormLoss()
        return lambda out, tgt: focal(out, tgt) + ln(out, tgt)

    elif loss_type == 'eim+focal':
        focal = FocalLoss(weight=weight)
        eim = EnhancedIsotropyLoss()
        return lambda out, tgt: focal(out, tgt) + 0.1 * eim(out)

    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

def train(args, model, enc=False):
    best_acc = 0

    #TODO: calculate weights by processing dataset histogram (now its being set by hand from the torch values)
    #create a loder to run all images and calculate histogram of labels, then create weight array using class balancing

    weight = torch.ones(NUM_CLASSES)
    if (enc and args.model == "erfnet"):
        weight[0] = 2.3653597831726	
        weight[1] = 4.4237880706787	
        weight[2] = 2.9691488742828	
        weight[3] = 5.3442072868347	
        weight[4] = 5.2983593940735	
        weight[5] = 5.2275490760803	
        weight[6] = 5.4394111633301	
        weight[7] = 5.3659925460815	
        weight[8] = 3.4170460700989	
        weight[9] = 5.2414722442627	
        weight[10] = 4.7376127243042	
        weight[11] = 5.2286224365234	
        weight[12] = 5.455126285553	
        weight[13] = 4.3019247055054	
        weight[14] = 5.4264230728149	
        weight[15] = 5.4331531524658	
        weight[16] = 5.433765411377	
        weight[17] = 5.4631009101868	
        weight[18] = 5.3947434425354
    elif not enc and args.model == "erfnet":
        weight[0] = 2.8149201869965	
        weight[1] = 6.9850029945374	
        weight[2] = 3.7890393733978	
        weight[3] = 9.9428062438965	
        weight[4] = 9.7702074050903	
        weight[5] = 9.5110931396484	
        weight[6] = 10.311357498169	
        weight[7] = 10.026463508606	
        weight[8] = 4.6323022842407	
        weight[9] = 9.5608062744141	
        weight[10] = 7.8698215484619	
        weight[11] = 9.5168733596802	
        weight[12] = 10.373730659485	
        weight[13] = 6.6616044044495	
        weight[14] = 10.260489463806	
        weight[15] = 10.287888526917	
        weight[16] = 10.289801597595	
        weight[17] = 10.405355453491	
        weight[18] = 10.138095855713
    elif args.model == "enet":
        weight = [
            7.8450451,
            3.36366406,
            14.04234086,
            4.9948856,
            39.25997007,
            36.5152765,
            32.90667927,
            46.27742179,
            40.67459427,
            6.71150498,
            33.5627786,
            18.54488148,
            32.99978951,
            47.68372067,
            12.70290829,
            45.20793195,
            45.7834263,
            45.82760469,
            48.40837536,
            42.76317799,
        ]
    elif args.model == "bisenetv1":
        weight =  [
            1.4297,
            1.4805,
            1.4363,
            3.365,
            2.6635,
            1.4311,
            2.1943,
            1.4817,
            1.4513,
            2.1984,
            1.5295,
            1.6892,
            3.2224,
            1.4727,
            7.5978,
            9.4117,
            15.2588,
            5.6818,
            2.2067,
            1,
        ]
    else:
        weight[0]  = 0.05749461494438303   # road
        weight[1]  = 0.33393575727571856   # sidewalk
        weight[2]  = 0.09300840061818595   # building
        weight[3]  = 1.069356175700148     # wall
        weight[4]  = 1.0670317701641996    # fence
        weight[5]  = 1.7366507242543063    # pole
        weight[6]  = 5.747970437863489     # traffic-light
        weight[7]  = 3.670816571454112     # traffic-sign
        weight[8]  = 0.1313745935016086    # vegetation
        weight[9]  = 1.0321980880219366    # terrain
        weight[10] = 0.4849822246648234    # sky
        weight[11] = 1.3910532474949333    # person
        weight[12] = 5.481691283227556     # rider
        weight[13] = 0.2924450717737152    # car
        weight[14] = 0.9697499025770556    # truck
        weight[15] = 0.8411727292719605    # bus
        weight[16] = 0.4404892260506257    # train
        weight[17] = 3.75885185752387      # motorcycle
        weight[18] = 2.8747245416888627    # bicycle
        weight[19] = 0.16401584690023238   # void / ignore	

    #weight[19] = 0

    assert os.path.exists(args.datadir), "Error: datadir (dataset directory) could not be loaded"

    if args.model == "erfnet":
        co_transform = MyCoTransform(enc, augment=True, height=args.height)
        co_transform_val = MyCoTransform(enc, augment=False, height=args.height)
    elif args.model == "enet" or args.model == "bisenetv1":
        co_transform = EnetCoTransform(augment=True, height=args.height)
        co_transform_val = EnetCoTransform(augment=False, height=args.height)
    
    dataset_train = cityscapes(args.datadir, co_transform, 'train')
    dataset_val = cityscapes(args.datadir, co_transform_val, 'val')
    
    '''
    #Calculate class weights
    freq = Counter()
    img_freq = Counter()
    for _, lbl in tqdm(dataset_train):   
        lbl = np.array(lbl)
        present = np.unique(lbl)
        for c in present:
            img_freq[c] += 1
            freq[c] += (lbl == c).sum()

    median = np.median([freq[c]/img_freq[c] for c in freq])
    weights = {c: median / (freq[c]/img_freq[c]) for c in freq}
    '''
    print("Class weights: ", weight)

    loader = DataLoader(dataset_train, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    if args.cuda:
      weight = weight.cuda()
    if args.model == "erfnet" or args.model == "enet":
        criterion = get_loss_function(args.loss_type, weight)
    elif args.model == "bisenetv1":
        criterion = get_loss_function(args.loss_type, weight=weight)
        criterion_aux1 = get_loss_function(args.loss_type, weight=weight)
        criterion_aux2 = get_loss_function(args.loss_type, weight=weight)
    print(f"Using loss: {args.loss_type}")

    savedir = f'../save/{args.savedir}'

    if (enc):
        automated_log_path = savedir + "/automated_log_encoder.txt"
        modeltxtpath = savedir + "/model_encoder.txt"
    else:
        automated_log_path = savedir + "/automated_log.txt"
        modeltxtpath = savedir + "/model.txt"    

    if (not os.path.exists(automated_log_path)):    #dont add first line if it exists 
        with open(automated_log_path, "a") as myfile:
            myfile.write("Epoch\t\tTrain-loss\t\tTest-loss\t\tTrain-IoU\t\tTest-IoU\t\tlearningRate")

    with open(modeltxtpath, "w") as myfile:
        myfile.write(str(model))


    #TODO: reduce memory in first gpu: https://discuss.pytorch.org/t/multi-gpu-training-memory-usage-in-balance/4163/4        #https://github.com/pytorch/pytorch/issues/1893

    scaler = GradScaler('cuda')
    #optimizer = Adam(model.parameters(), 5e-4, (0.9, 0.999),  eps=1e-08, weight_decay=2e-4)     ## scheduler 1
    if args.model == "enet":
        optimizer = Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999), weight_decay=0.0002)
    elif args.model == "erfnet":
        optimizer = Adam(model.parameters(), 5e-4, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4)      ## scheduler 2
    elif args.model == "bisenetv1":
        optimizer = SGD(model.parameters(), lr=2.5e-2, momentum=0.9, weight_decay=1e-4)

    start_epoch = 1
    if args.resume:
        #Must load weights, optimizer, epoch and best value. 
        if enc:
            filenameCheckpoint = savedir + '/checkpoint_enc.pth.tar'
        else:
            filenameCheckpoint = savedir + '/checkpoint.pth.tar'

        assert os.path.exists(filenameCheckpoint), "Error: resume option was used but checkpoint was not found in folder"
        checkpoint = torch.load(filenameCheckpoint)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint['best_acc']
        print("=> Loaded checkpoint at epoch {})".format(checkpoint['epoch']))

    #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5) # set up scheduler     ## scheduler 1
    if(args.model == "erfnet"):
        lambda1 = lambda epoch: pow((1-((epoch-1)/args.num_epochs)),0.9)  ## scheduler 2
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1) ## scheduler 2
    elif(args.model == "enet" or args.model == "bisenetv1"):
        max_iter = args.num_epochs * len(loader)
        poly = lambda it: (1 - it/max_iter) ** 0.9
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, poly)

    if args.visualize and args.steps_plot > 0:
        board = Dashboard(args.port)

    accum_iter = 2 
    for epoch in range(start_epoch, args.num_epochs+1):
        print("----- TRAINING - EPOCH", epoch, "-----")

        #scheduler.step(epoch)    ## scheduler 2

        epoch_loss = []
        time_train = []
     
        doIouTrain = args.iouTrain   
        doIouVal =  args.iouVal      

        if (doIouTrain):
            iouEvalTrain = iouEval(NUM_CLASSES)

        usedLr = 0
        for param_group in optimizer.param_groups:
            print("LEARNING RATE: ", param_group['lr'])
            usedLr = float(param_group['lr'])

        model.train()
        for step, (images, labels) in enumerate(loader):

            start_time = time.time()
            #print (labels.size())
            #print (np.unique(labels.numpy()))
            #print("labels: ", np.unique(labels[0].numpy()))
            #labels = torch.ones(4, 1, 512, 1024).long()
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()

            inputs = Variable(images)
            targets = Variable(labels)
            with autocast('cuda'):
                if args.model == "erfnet":
                    outputs = model(inputs, only_encode=enc)
                elif args.model == "enet":
                    outputs = model(inputs)
                elif args.model == "bisenetv1":
                    outputs, aux1, aux2 = model(inputs)
                    loss1 = criterion(outputs, targets[:, 0])
                    loss2 = criterion_aux1(aux1, targets[:, 0])
                    loss3 = criterion_aux2(aux2, targets[:, 0])
                    loss  = loss1 + 0.4*(loss2+loss3)
                if args.model != "bisenetv1":
                    loss = criterion(outputs, targets[:,0])

            loss = loss / accum_iter
            scaler.scale(loss).backward()

            if (step + 1) % accum_iter == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()  # scheduler step after optimizer step

            epoch_loss.append(loss.data.item())
            time_train.append(time.time() - start_time)

            if (doIouTrain):
                #start_time_iou = time.time()
                iouEvalTrain.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)
                #print ("Time to add confusion matrix: ", time.time() - start_time_iou)      

            #print(outputs.size())
            if args.visualize and args.steps_plot > 0 and step % args.steps_plot == 0:
                start_time_plot = time.time()
                image = inputs[0].cpu().data
                #image[0] = image[0] * .229 + .485
                #image[1] = image[1] * .224 + .456
                #image[2] = image[2] * .225 + .406
                #print("output", np.unique(outputs[0].cpu().max(0)[1].data.numpy()))
                board.image(image, f'input (epoch: {epoch}, step: {step})')
                if isinstance(outputs, list):   #merge gpu tensors
                    board.image(color_transform(outputs[0][0].cpu().max(0)[1].data.unsqueeze(0)),
                    f'output (epoch: {epoch}, step: {step})')
                else:
                    board.image(color_transform(outputs[0].cpu().max(0)[1].data.unsqueeze(0)),
                    f'output (epoch: {epoch}, step: {step})')
                board.image(color_transform(targets[0].cpu().data),
                    f'target (epoch: {epoch}, step: {step})')
                print ("Time to paint images: ", time.time() - start_time_plot)
            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss) / len(epoch_loss)
                print(f'loss: {average:0.4} (epoch: {epoch}, step: {step})', 
                        "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size))

            
        average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
        
        iouTrain = 0
        if (doIouTrain):
            iouTrain, iou_classes = iouEvalTrain.getIoU()
            iouStr = getColorEntry(iouTrain)+'{:0.2f}'.format(iouTrain*100) + '\033[0m'
            print ("EPOCH IoU on TRAIN set: ", iouStr, "%")  

        #Validate on 500 val images after each epoch of training
        print("----- VALIDATING - EPOCH", epoch, "-----")
        model.eval()
        epoch_loss_val = []
        time_val = []

        if (doIouVal):
            iouEvalVal = iouEval(NUM_CLASSES)

        for step, (images, labels) in enumerate(loader_val):
            start_time = time.time()
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()

            inputs = Variable(images, volatile=True)    #volatile flag makes it free backward or outputs for eval
            targets = Variable(labels, volatile=True)
            if args.model == "erfnet":
                outputs = model(inputs, only_encode=enc)
                loss = criterion(outputs, targets[:, 0])
            elif args.model == "enet":
                outputs = model(inputs)
                loss = criterion(outputs, targets[:, 0])
            elif args.model == "bisenetv1":
                outputs, aux1, aux2 = model(inputs)
                loss1 = criterion(outputs, targets[:, 0])
                loss2 = criterion_aux1(aux1, targets[:, 0])
                loss3 = criterion_aux2(aux2, targets[:, 0])
                loss = loss1 + 0.4 * (loss2 + loss3)  # Pesi delle perdite ausiliarie

            epoch_loss_val.append(loss.data.item())
            time_val.append(time.time() - start_time)


            #Add batch to calculate TP, FP and FN for iou estimation
            if (doIouVal):
                #start_time_iou = time.time()
                iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)
                #print ("Time to add confusion matrix: ", time.time() - start_time_iou)

            if args.visualize and args.steps_plot > 0 and step % args.steps_plot == 0:
                start_time_plot = time.time()
                image = inputs[0].cpu().data
                board.image(image, f'VAL input (epoch: {epoch}, step: {step})')
                if isinstance(outputs, list):   #merge gpu tensors
                    board.image(color_transform(outputs[0][0].cpu().max(0)[1].data.unsqueeze(0)),
                    f'VAL output (epoch: {epoch}, step: {step})')
                else:
                    board.image(color_transform(outputs[0].cpu().max(0)[1].data.unsqueeze(0)),
                    f'VAL output (epoch: {epoch}, step: {step})')
                board.image(color_transform(targets[0].cpu().data),
                    f'VAL target (epoch: {epoch}, step: {step})')
                print ("Time to paint images: ", time.time() - start_time_plot)
            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss_val) / len(epoch_loss_val)
                print(f'VAL loss: {average:0.4} (epoch: {epoch}, step: {step})', 
                        "// Avg time/img: %.4f s" % (sum(time_val) / len(time_val) / args.batch_size))
                       

        average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)
        #scheduler.step(average_epoch_loss_val, epoch)  ## scheduler 1   # update lr if needed

        iouVal = 0
        if (doIouVal):
            iouVal, iou_classes = iouEvalVal.getIoU()
            iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
            print ("EPOCH IoU on VAL set: ", iouStr, "%") 
           

        # remember best valIoU and save checkpoint
        if iouVal == 0:
            current_acc = -average_epoch_loss_val
        else:
            current_acc = iouVal 
        is_best = current_acc > best_acc
        best_acc = max(current_acc, best_acc)
        if enc:
            filenameCheckpoint = savedir + '/checkpoint_enc.pth.tar'
            filenameBest = savedir + '/model_best_enc.pth.tar'    
        else:
            filenameCheckpoint = savedir + '/checkpoint.pth.tar'
            filenameBest = savedir + '/model_best.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filenameCheckpoint, filenameBest)

        #SAVE MODEL AFTER EPOCH
        if (enc):
            filename = f'{savedir}/model_encoder-{epoch:03}.pth'
            filenamebest = f'{savedir}/model_encoder_best.pth'
        else:
            filename = f'{savedir}/model-{epoch:03}.pth'
            filenamebest = f'{savedir}/model_best.pth'
        if args.epochs_save > 0 and step > 0 and step % args.epochs_save == 0:
            torch.save(model.state_dict(), filename)
            print(f'save: {filename} (epoch: {epoch})')
        if (is_best):
            torch.save(model.state_dict(), filenamebest)
            print(f'save: {filenamebest} (epoch: {epoch})')
            if (not enc):
                with open(savedir + "/best.txt", "w") as myfile:
                    myfile.write("Best epoch is %d, with Val-IoU= %.4f" % (epoch, iouVal))   
            else:
                with open(savedir + "/best_encoder.txt", "w") as myfile:
                    myfile.write("Best epoch is %d, with Val-IoU= %.4f" % (epoch, iouVal))           

        #SAVE TO FILE A ROW WITH THE EPOCH RESULT (train loss, val loss, train IoU, val IoU)
        #Epoch		Train-loss		Test-loss	Train-IoU	Test-IoU		learningRate
        with open(automated_log_path, "a") as myfile:
            myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.8f" % (epoch, average_epoch_loss_train, average_epoch_loss_val, iouTrain, iouVal, usedLr ))
    
    return(model)   #return model (convenience for encoder-decoder training)

def save_checkpoint(state, is_best, filenameCheckpoint, filenameBest):
    torch.save(state, filenameCheckpoint)
    if is_best:
        print ("Saving model as best")
        torch.save(state, filenameBest)


def main(args):
    savedir = f'../save/{args.savedir}'

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    with open(savedir + '/opts.txt', "w") as myfile:
        myfile.write(str(args))

    #Load Model
    assert os.path.exists(args.model + ".py"), "Error: model definition not found"
    model_file = importlib.import_module(args.model)
    model = model_file.Net(NUM_CLASSES)
    model = model.to(memory_format=torch.channels_last)
    copyfile(args.model + ".py", savedir + '/' + args.model + ".py")
    
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()
    
    if args.state:
        #if args.state is provided then load this state for training
        #Note: this only loads initialized weights. If you want to resume a training use "--resume" option!!
        """
        try:
            model.load_state_dict(torch.load(args.state))
        except AssertionError:
            model.load_state_dict(torch.load(args.state,
                map_location=lambda storage, loc: storage))
        #When model is saved as DataParallel it adds a model. to each key. To remove:
        #state_dict = {k.partition('model.')[2]: v for k,v in state_dict}
        #https://discuss.pytorch.org/t/prefix-parameter-names-in-saved-model-if-trained-by-multi-gpu/494
        """
        def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict keys are there
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                     continue
                own_state[name].copy_(param)
            return model

        #print(torch.load(args.state))
        model = load_my_state_dict(model, torch.load(args.state))

    """
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            #m.weight.data.normal_(0.0, 0.02)
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif classname.find('BatchNorm') != -1:
            #m.weight.data.normal_(1.0, 0.02)
            m.weight.data.fill_(1)
            m.bias.data.fill_(0)

    #TO ACCESS MODEL IN DataParallel: next(model.children())
    #next(model.children()).decoder.apply(weights_init)
    #Reinitialize weights for decoder
    
    next(model.children()).decoder.layers.apply(weights_init)
    next(model.children()).decoder.output_conv.apply(weights_init)

    #print(model.state_dict())
    f = open('weights5.txt', 'w')
    f.write(str(model.state_dict()))
    f.close()
    """

    #train(args, model)
    if (not args.decoder and  args.model == "erfnet"):
        print("========== ENCODER TRAINING ===========")
        model = train(args, model, True) #Train encoder
        #CAREFUL: for some reason, after training encoder alone, the decoder gets weights=0. 
        #We must reinit decoder weights or reload network passing only encoder in order to train decoder
        print("========== DECODER TRAINING ===========")
        if (not args.state):
            if args.pretrainedEncoder:
                print("Loading encoder pretrained in imagenet")
                from erfnet_imagenet import ERFNet as ERFNet_imagenet
                pretrainedEnc = torch.nn.DataParallel(ERFNet_imagenet(1000))
                pretrainedEnc.load_state_dict(torch.load(args.pretrainedEncoder)['state_dict'])
                pretrainedEnc = next(pretrainedEnc.children()).features.encoder
                if (not args.cuda):
                    pretrainedEnc = pretrainedEnc.cpu()     #because loaded encoder is probably saved in cuda
            else:
                pretrainedEnc = next(model.children()).encoder
            model = model_file.Net(NUM_CLASSES, encoder=pretrainedEnc)  #Add decoder to encoder
            if args.cuda:
                model = torch.nn.DataParallel(model).cuda()
        #When loading encoder reinitialize weights for decoder because they are set to 0 when training dec
    model = train(args, model, False)   #Train decoder
    print("========== TRAINING FINISHED ===========")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True)  #NOTE: cpu-only has not been tested so you might have to change code if you deactivate this flag
    parser.add_argument('--model', default="erfnet", help='Model to use: erfnet | enet | bisenetv1')
    parser.add_argument('--state')
    parser.add_argument('--port', type=int, default=8097)
    parser.add_argument('--datadir', default=os.getenv("HOME") + "/datasets/cityscapes/", help='Path to dataset directory')
    #parser.add_argument('--datadir', default="C:/Users/nikde/Desktop/MAGISTRALE/SECONDO ANNO/ADVANCE MACHINE/PROJECT/Cityscapes")
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--num-epochs', type=int, default=150)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--steps-loss', type=int, default=50)
    parser.add_argument('--steps-plot', type=int, default=50)
    parser.add_argument('--epochs-save', type=int, default=0)    #You can use this value to save model every X epochs
    parser.add_argument('--savedir', required=True)
    parser.add_argument('--decoder', action='store_true')
    parser.add_argument('--pretrainedEncoder') #, default="../trained_models/erfnet_encoder_pretrained.pth.tar")
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--loss-type', type=str, default='crossentropy',
                    help='Type of loss function: crossentropy | focal | logitnorm | isotropy | logitnorm+ce | isotropy+focal | ecc.')
    parser.add_argument('--iouTrain', action='store_true', default=False) #recommended: False (takes more time to train otherwise)
    parser.add_argument('--iouVal', action='store_true', default=True)  
    parser.add_argument('--resume', action='store_true')    #Use this flag to load last checkpoint for training  

    main(parser.parse_args()) 
