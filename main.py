import os
import torch
import torchvision 
import torch.nn as nn
import torchvision.transforms as transforms
from dataset_minst import *
import time
import numpy as np
import argparse
from mnist_reader import *
from dataset_minst import *
import pdb
from datetime import datetime
from confusionmeter import *
import timm
#https://blog.csdn.net/idwtwt/article/details/87625377


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(1, 3, kernel_size=1)
        self.resnet = torchvision.models.resnet50(pretrained=True)
 
    def forward(self, x):
 
        x= self.conv(x)
        x= self.resnet(x)
        return x

parser = argparse.ArgumentParser(description='PyTorch CNN MNIST Learning') # 创建ArgumentParser对象
parser.add_argument('--cls-num', default=10, type=int, metavar='N', help='class number')
parser.add_argument('--epoch', default=200, type=int, metavar='N', help='epochs')

parser.add_argument('--dataroot', default='/data1/scz/data/MNIST/train.csv', type=str, help='data path')
parser.add_argument('--bs', default=192, type=int,metavar='N', help='batch size')
parser.add_argument('--imsize', default=224, type=int,metavar='N', help='image size')
parser.add_argument('--lr', default=3e-4, type=float,metavar='N', help='learning rate')

parser.add_argument('--net', default='resnet50', type=str,help='network')
parser.add_argument('--loss', default='cross', type=str,help='loss function')

parser.add_argument('--multi-gpu',action='store_true',help='use multi gpu?')

def main():
    global args
    args = parser.parse_args()

    cls_num = args.cls_num
    EPOCHS = args.epoch
    BATCH_SIZE = args.bs
    image_size = args.imsize
    lr = args.lr
    dataroot = args.dataroot

    net_name = args.net
    loss_name = args.loss

    model = torchvision.models.resnet50(pretrained=True)
    #model = timm.create_model('resnet50', pretrained=True, num_classes=cls_num, in_chans=1)
    #model =  Net().cuda()
    model.fc = nn.Linear(in_features=2048, out_features=cls_num, bias=True)
    device = torch.device('cuda')
    print("Using", torch.cuda.device_count(), "GPUs!")
    criterion =  nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.5)

    model.cuda()
    if args.multi_gpu:
        model=nn.DataParallel(model,device_ids=[0,1,2,3])

    transform=transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

    mnist_train = MNIST(path = dataroot, train = True, transform = transform)
    mnist_test = MNIST(path = dataroot, train = False, transform = transform)
    
    mnist_train_loader = torch.utils.data.DataLoader(
            dataset=mnist_train, batch_size=BATCH_SIZE, shuffle=True, 
            num_workers = 4, pin_memory=True, sampler=None,)

    mnist_test_loader = torch.utils.data.DataLoader(
            dataset=mnist_test, batch_size=BATCH_SIZE, shuffle=True, 
            num_workers = 4, pin_memory=True, sampler=None,)
    
    localtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    d = localtime+'_'+args.loss;
    code_loc = '/data1/scz/MNIST/Train_model_result'
    directory = os.path.join(code_loc,d)
    print('\ndirectory:',directory,'\n')

    if not os.path.exists(directory):
        os.makedirs(directory)
    Logger_file = os.path.join(directory,"log.txt")

    best_prec = 0

    for epoch in range(EPOCHS):
        torch.cuda.empty_cache()
        train(mnist_train_loader,model, epoch, criterion, optimizer, args, criterion_metric=None)
        torch.cuda.empty_cache()
        acc,avg_prec,avg_recall = test(mnist_test_loader, model, epoch, criterion)
        
        print("Current Time =", datetime.now().strftime("%H:%M:%S"))
        with open(Logger_file,'a') as f:
            f.write("epoch:{}\tAccuracy:{}\tAverage precision:{}\tAverage recall:{}\n".format(epoch+1,acc,avg_prec,avg_recall))
        
        if avg_prec >= best_prec:
            print('Best model saved!')
            best_prec = avg_prec
            path = os.path.join(directory,'best_model.pth')
            torch.save(model,path)

def train(train_loader, model, epoch, criterion, optimizer, args, criterion_metric=None):
    batch_time = AverageMeter()
    end = time.time()
    model.train()
    losses = AverageMeter()

    for step, (x, y) in enumerate(train_loader):
        batch_time.update(time.time() - end)
        # x = x.squeeze()
        # y = y.squeeze()
        x = x.cuda()
        out = model(x)
        loss = criterion(out, y.long().cuda())
        losses.update(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0 and step != 0:
            print('>> Train: [{0}][{1}/{2}]\tlr:{3}\t'
                'losses {losses.val:.3f} ({losses.avg:.3f})\t'                
                .format(
                epoch+1, step+1, len(train_loader), optimizer.param_groups[0]['lr'], 
                batch_time=batch_time, losses = losses
                ))
    Loss_file = os.path.join('/data1/scz/MNIST/Train_model_result',"loss.txt")
    with open(Loss_file,'a') as f:
        f.write('Epoch: {}\tLoss {losses.avg:.4f}\n'.format(epoch+1,losses = losses))

def test(test_loader, model, epoch, criterion):
    print('>> Evaluating network on test datasets...')
    batch_time = AverageMeter()
    end = time.time()
    model.eval()
    confusion_matrix = ConfusionMeter(10)

    for step, (x, y) in enumerate(test_loader):
        batch_time.update(time.time() - end)
        end = time.time()
        x = x.cuda()
        x = x.contiguous()
        y = y.cuda()
        with torch.no_grad():
            out = model(x) 
        confusion_matrix.add(out.data, y.data)

        if step % 100 == 0:
            print('>> Test: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                .format(
                epoch+1, step+1, len(test_loader), batch_time=batch_time,
                ))
    cm_value = confusion_matrix.value()

    acc = 100. * (cm_value[0][0] + cm_value[1][1] + cm_value[2][2] + cm_value[3][3] + cm_value[4][4] + cm_value[5][5] + cm_value[6][6] + cm_value[7][7] + cm_value[8][8] + cm_value[9][9]) / (cm_value.sum())
    def prec(cm_value,n):
        col_sum = cm_value.sum(axis=0)
        if cm_value[n][n] !=0 or col_sum[n] != 0:
            return cm_value[n][n]/col_sum[n]
        else:
            return 0
    def recall(cm_value,n):
        row_sum = cm_value.sum(axis=1)
        if cm_value[n][n] !=0 or row_sum[n] != 0:
            return cm_value[n][n]/row_sum[n]
        else:
            return 0

    avg_prec = 100. * (prec(cm_value,0) + prec(cm_value,1) + prec(cm_value,2) + prec(cm_value,3) + prec(cm_value,4) + prec(cm_value,5) + prec(cm_value,6) + prec(cm_value,7) + prec(cm_value,8) + prec(cm_value,9)) / 10

    avg_recall = 100. *(recall(cm_value,0) + recall(cm_value,1) + recall(cm_value,2) + recall(cm_value,3) + recall(cm_value,4) + recall(cm_value,5) + recall(cm_value,6) + recall(cm_value,7) + recall(cm_value,8) + recall(cm_value,9)) / 10
    
    print("confusion_matrix",acc, "avg_prec:",avg_prec, "avg_recall:",avg_recall)
    
    return acc,avg_prec,avg_recall

if __name__=='__main__':
    main()
    #CUDA_VISIBLE_DEVICES='4' python main.py