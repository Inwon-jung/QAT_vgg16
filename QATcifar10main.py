import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision as tv
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import argparse

from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from QATcifar10 import *


'''
model_path = './model_pth/vgg16_bn-6c64b313.pth'  
BATCH_SIZE = 128
LR = 0.01  # learning rate
EPOCH = 10
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
'''

class Net(nn.Module):
    def __init__(self,num_classes=10):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,64,3,1,1)
        self.conv2 = nn.Conv2d(64,64,3,1,1)
        self.conv3 = nn.Conv2d(64,128,3,1,1)
        self.conv4 = nn.Conv2d(128,128,3,1,1)
        self.conv5 = nn.Conv2d(128,256,3,1,1)
        self.conv6 = nn.Conv2d(256,256,3,1,1)
        self.conv7 = nn.Conv2d(256,256,3,1,1)
        self.conv8 = nn.Conv2d(256,512,3,1,1)
        self.conv9 = nn.Conv2d(512,512,3,1,1)
        self.conv10 = nn.Conv2d(512,512,3,1,1)
        self.conv11 = nn.Conv2d(512,512,3,1,1)
        self.conv12 = nn.Conv2d(512,512,3,1,1)
        self.conv13 = nn.Conv2d(512,512,3,1,1)
        
        self.ReLU = nn.ReLU(inplace = True)
        self.Maxpool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(512,10)
        
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(256)
        self.bn8 = nn.BatchNorm2d(512)
        self.bn9 = nn.BatchNorm2d(512)
        self.bn10 = nn.BatchNorm2d(512)
        self.bn11 = nn.BatchNorm2d(512)
        self.bn12 = nn.BatchNorm2d(512)
        self.bn13 = nn.BatchNorm2d(512)
        
        
    def forward(self,x):
        x = self.ReLU(self.conv1(x))
        x = self.bn1(x)
        x = self.ReLU(self.conv2(x))
        x = self.bn2(x)
        x = self.Maxpool(x)
        x = self.ReLU(self.conv3(x))
        x = self.bn3(x)
        x = self.ReLU(self.conv4(x))
        x = self.bn4(x)
        x = self.Maxpool(x)
        x = self.ReLU(self.conv5(x))
        x = self.bn5(x)
        x = self.ReLU(self.conv6(x))
        x = self.bn6(x)
        x = self.ReLU(self.conv7(x))
        x = self.bn7(x)
        x = self.Maxpool(x)
        x = self.ReLU(self.conv8(x))
        x = self.bn8(x)
        x = self.ReLU(self.conv9(x))
        x = self.bn9(x)
        x = self.ReLU(self.conv10(x))
        x = self.bn10(x)
        x = self.Maxpool(x)
        x = self.ReLU(self.conv11(x))
        x = self.bn11(x)
        x = self.ReLU(self.conv12(x))
        x = self.bn12(x)
        x = self.ReLU(self.conv13(x))
        x = self.bn13(x)
        x = self.Maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        #return F.log_softmax(x, dim=1)
        return x

def visualise(x, axs):
    x = x.view(-1).cpu().numpy()
    axs.hist(x)


def train(args, model, device, train_loader, optimizer, epoch, test_loader):
    model.train()
    for batch_idx, (data,target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output,target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % args["log_interval"] == 0:
            print('Train Epoch: {} [{}/{} ({:0f}%]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),loss.item()
            ))
            test(args, model, device, test_loader)
            
def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction = 'sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    test_loss /= len(test_loader.dataset)
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    batch_size = 128
    test_batch_size = 128
    epochs = 10
    lr = 0.01
    momentum = 0.9
    input_size = 32
    seed = 1
    log_interval = 39
    save_model = False
    no_cuda = False
    dataset_root = 'F:\Torchcode\Binarized-Neural-networks-using-pytorch-master\Cifar10\data\cifar-10-batches-py'
    
    use_cuda = not no_cuda and torch.cuda.is_available()

    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    kwargs = {'num_workers':0, 'pin_memory': True} if use_cuda else {}

    transform = transforms.Compose(
        [
         transforms.Resize(input_size),
         transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    )     
    trainset = datasets.CIFAR10(root = dataset_root, train=True,
                                download = True, transform = transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size = batch_size,
                                               shuffle = True, num_workers=0)
    
    testset = datasets.CIFAR10(root = dataset_root, train = False,
                               download = True, transform = transform)
    test_loader = torch.utils.data.DataLoader(testset,batch_size = test_batch_size,
                                              shuffle = False, num_workers=0)
    
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr = lr, momentum = momentum)
    args = {}
    args["log_interval"] = log_interval
    for epoch in range(1, epochs+1):
        train(args, model, device, train_loader, optimizer, epoch, test_loader)
        
        test(args, model, device, test_loader)
        
    if(save_model):
        torch.save(model.state_dict(), "cifar_cnn_test.pt")
        
    return model

#model = main()

# Quantization Aware Training Forward Pass
def quantAwareTrainingForward(model, x, stats, vis = False, axs = None, sym = False, num_bits = 8, act_quant = False):
    
    conv1weight = model.conv1.weight.data
    model.conv1.weight.data = FakeQuantOp.apply(model.conv1.weight.data, num_bits)
    x = F.relu(model.conv1(x))
    x = model.bn1(x)

    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1),stats, 'conv1')
        
    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['conv1']['ema_min'], stats['conv1']['ema_max'])
        
    conv2weight = model.conv2.weight.data
    model.conv2.weight.data = FakeQuantOp.apply(model.conv2.weight.data, num_bits)
    x = F.relu(model.conv2(x))
    x = model.bn2(x)
    
    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1),stats, 'conv2')
        
    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['conv2']['ema_min'], stats['conv2']['ema_max'])
        
    x = F.max_pool2d(x,2,2)
    
    
    conv3weight = model.conv3.weight.data
    model.conv3.weight.data = FakeQuantOp.apply(model.conv3.weight.data, num_bits)
    x = F.relu(model.conv3(x))
    x = model.bn3(x)
    
    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1),stats, 'conv3')
        
    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['conv3']['ema_min'], stats['conv3']['ema_max'])        
        
        
    conv4weight = model.conv4.weight.data
    model.conv4.weight.data = FakeQuantOp.apply(model.conv4.weight.data, num_bits)
    x = F.relu(model.conv4(x))
    x = model.bn4(x)
    
    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1),stats, 'conv4')
        
    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['conv4']['ema_min'], stats['conv4']['ema_max'])
        
    x = F.max_pool2d(x,2,2)
    
    conv5weight = model.conv5.weight.data
    model.conv5.weight.data = FakeQuantOp.apply(model.conv5.weight.data, num_bits)
    x = F.relu(model.conv5(x))
    x = model.bn5(x)
    
    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1),stats, 'conv5')
        
    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['conv5']['ema_min'], stats['conv5']['ema_max'])
        
    conv6weight = model.conv6.weight.data
    model.conv6.weight.data = FakeQuantOp.apply(model.conv6.weight.data, num_bits)
    x = F.relu(model.conv6(x))
    x = model.bn6(x)
    
    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1),stats, 'conv6')
        
    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['conv6']['ema_min'], stats['conv6']['ema_max'])

    conv7weight = model.conv7.weight.data
    model.conv7.weight.data = FakeQuantOp.apply(model.conv7.weight.data, num_bits)
    x = F.relu(model.conv7(x))
    x = model.bn7(x)
    
    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1),stats, 'conv7')
        
    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['conv7']['ema_min'], stats['conv7']['ema_max'])      
        
    x = F.max_pool2d(x,2,2)
        
    conv8weight = model.conv8.weight.data
    model.conv8.weight.data = FakeQuantOp.apply(model.conv8.weight.data, num_bits)
    x = F.relu(model.conv8(x))
    x = model.bn8(x)
    
    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1),stats, 'conv8')
        
    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['conv8']['ema_min'], stats['conv8']['ema_max'])
        
    conv9weight = model.conv9.weight.data
    model.conv9.weight.data = FakeQuantOp.apply(model.conv9.weight.data, num_bits)
    x = F.relu(model.conv9(x))
    x = model.bn9(x)
    
    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1),stats, 'conv9')
        
    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['conv9']['ema_min'], stats['conv9']['ema_max'])        
        
    conv10weight = model.conv10.weight.data
    model.conv10.weight.data = FakeQuantOp.apply(model.conv10.weight.data, num_bits)
    x = F.relu(model.conv10(x))
    x = model.bn10(x)
    
    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1),stats, 'conv10')
        
    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['conv10']['ema_min'], stats['conv10']['ema_max'])        
        
    x = F.max_pool2d(x,2,2)
    
    conv11weight = model.conv11.weight.data
    model.conv11.weight.data = FakeQuantOp.apply(model.conv11.weight.data, num_bits)
    x = F.relu(model.conv11(x))
    x = model.bn11(x)
    
    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1),stats, 'conv11')
        
    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['conv11']['ema_min'], stats['conv11']['ema_max'])        
        
    conv12weight = model.conv12.weight.data
    model.conv12.weight.data = FakeQuantOp.apply(model.conv12.weight.data, num_bits)
    x = F.relu(model.conv12(x))
    x = model.bn12(x)
    
    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1),stats, 'conv12')
        
    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['conv12']['ema_min'], stats['conv12']['ema_max'])        
        
    conv13weight = model.conv13.weight.data
    model.conv13.weight.data = FakeQuantOp.apply(model.conv13.weight.data, num_bits)
    x = F.relu(model.conv13(x))
    x = model.bn13(x)
    
    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1),stats, 'conv13')
        
    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['conv13']['ema_min'], stats['conv13']['ema_max'])     
    
    x = F.max_pool2d(x,2,2)
    
    x = x.view(x.size(0),-1)
    
    x = model.fc1(x)
    
    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0],-1),stats, 'fc1')
        
    return x, \
        conv1weight, conv2weight, conv3weight, conv4weight, conv5weight, conv6weight, conv7weight, \
            conv8weight, conv9weight, conv10weight, conv11weight, conv12weight, conv13weight,\
                stats
                
# Train using Quantization Aware Training

def trainQuantAware(args, model, device, train_loader,test_loader, optimizer, epoch, stats, act_quant=False, num_bits = 8):
    model.train()

    for batch_idx, (data,target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, \
            conv1weight, conv2weight, conv3weight, conv4weight, conv5weight, conv6weight, conv7weight,\
                conv8weight, conv9weight, conv10weight, conv11weight, conv12weight, conv13weight,\
                    stats = quantAwareTrainingForward(model,data, stats, num_bits=num_bits, act_quant = act_quant)
                    
    
    
        model.conv1.weight.data = conv1weight
        model.conv2.weight.data = conv2weight
        model.conv3.weight.data = conv3weight
        model.conv4.weight.data = conv4weight
        model.conv5.weight.data = conv5weight
        model.conv6.weight.data = conv6weight
        model.conv7.weight.data = conv7weight
        model.conv8.weight.data = conv8weight
        model.conv9.weight.data = conv9weight
        model.conv10.weight.data = conv10weight
        model.conv11.weight.data = conv11weight
        model.conv12.weight.data = conv12weight
        model.conv13.weight.data = conv13weight
        loss = F.cross_entropy(output,target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % args["log_interval"] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100.*batch_idx / len(train_loader),loss.item()))
            
            testQuantAware(args, model, device, test_loader, stats, act_quant, num_bits = num_bits)
        
    return stats


def testQuantAware(args, model, device, test_loader, stats, act_quant, num_bits = 4):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, \
                conv1weight, conv2weight, conv3weight, conv4weight, conv5weight, conv6weight, conv7weight,\
                    conv8weight, conv9weight, conv10weight, conv11weight, conv12weight, conv13weight,\
                        stats = quantAwareTrainingForward(model,data,stats,num_bits=num_bits,act_quant=act_quant)
                        
            model.conv1.weight.data = conv1weight
            model.conv2.weight.data = conv2weight
            model.conv3.weight.data = conv3weight
            model.conv4.weight.data = conv4weight
            model.conv5.weight.data = conv5weight
            model.conv6.weight.data = conv6weight
            model.conv7.weight.data = conv7weight
            model.conv8.weight.data = conv8weight
            model.conv9.weight.data = conv9weight
            model.conv10.weight.data = conv10weight
            model.conv11.weight.data = conv11weight
            model.conv12.weight.data = conv12weight
            model.conv13.weight.data = conv13weight

            test_loss += F.cross_entropy(output,target, reduction='sum').item() #sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log_probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    test_loss /= len(test_loader.dataset)
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100.* correct / len(test_loader.dataset)))

    

    
def mainQuantAware():
    print(mainQuantAware.__name__)
    
    batch_size = 128
    test_batch_size = 128
    epochs = 10
    lr = 0.01
    momentum = 0.9
    input_size = 32
    seed = 1
    log_interval = 39
    save_model = False
    no_cuda = False
    stat_QAT_epoch = 4
    num_bits = 8
    dataset_root = 'F:\Torchcode\Binarized-Neural-networks-using-pytorch-master\Cifar10\data\cifar-10-batches-py'
    
    
    use_cuda = not no_cuda and torch.cuda.is_available()
    
    torch.manual_seed(seed)
    
    device = torch.device("cuda" if use_cuda else "cpu")
    
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    
    transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    )

    trainset = datasets.CIFAR10(root=dataset_root, train=True,
                                download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=0)

    testset = datasets.CIFAR10(root=dataset_root, train=False,
                               download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                              shuffle=False, num_workers=0)


    
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr = lr, momentum = momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size =10, gamma = 0.1)
    args = {}
    args["log_interval"] = log_interval
    
    stats = {}
    
    for epoch in range(1, epochs + 1):
        if epoch > stat_QAT_epoch:
            act_quant = True
            
        else:
            act_quant = False
            
        stats = trainQuantAware(args, model, device, train_loader, test_loader, optimizer, epoch, stats, act_quant, 
                                num_bits = num_bits)
        scheduler.step()
        
        testQuantAware(args, model, device, test_loader, stats, act_quant, num_bits=num_bits)
        
    if (save_model):
        torch.save(model.state_dict(), "cifar_cnn.pt")
        
    return model, stats


# model = main()

model, old_stats = mainQuantAware()
            
            
            
            
            
    
    
    
    
    
    
    
    
    
    
    
    
    