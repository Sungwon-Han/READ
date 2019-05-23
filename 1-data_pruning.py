import os
import csv
import glob
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from torch.utils.data import DataLoader
from skimage import io, transform
from dataloader import *
from parameters import *

def main(args):
    LOAD = args.load
    LOAD_PATH = args.modelurl
    train_gps = RemovalDataset(metadata = "./data/proxy/is_environment_train.csv", root_dir = "./data/proxy/train/",
                         transform=transforms.Compose([
                             RandomRotate(),
                             ToTensor(),
                             Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                         ]))

    test_gps = RemovalDataset(metadata = "./data/proxy/is_environment_test.csv", root_dir = "./data/proxy/test/",
                            transform=transforms.Compose([
                                 ToTensor(),
                                 Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ]))

    train_loader = torch.utils.data.DataLoader(train_gps, batch_size=args.batch_size, shuffle=True, num_workers = args.workers)
    test_loader = torch.utils.data.DataLoader(test_gps, batch_size=len(test_gps), shuffle=False, num_workers = args.workers) 

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)    
       
    if LOAD == False:           
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        cudnn.benchmark = True
        for epoch in range(args.epochs):
            train(train_loader, model, optimizer, epoch, args.batch_size)
            if (epoch + 1) % args.evaluation_epochs == 0:
                validate(test_loader, model)

            if (epoch + 1) % args.checkpoint_epochs == 0:
                save_checkpoint({'state_dict': model.state_dict()}, "./model", epoch + 1)
    else:           
        model.load_state_dict(torch.load(LOAD_PATH)['state_dict'], strict=False)
        model.to(device)
        print("Load Finished")       
        
    model.eval()
    remove_environment(model, args.train_path)
    remove_environment(model, args.test_path)


def train(train_loader, model, optimizer, epoch, batch_size):
    criterion = nn.BCELoss()
    model.train()
    total_loss = 0
    count = 0
    for i_batch, sample_batched in enumerate(train_loader):
        input_image = torch.autograd.Variable(sample_batched['image'].cuda())
        t = torch.autograd.Variable(sample_batched['y'].cuda()).float()
        y = model(input_image)
        loss = criterion(y, t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        count += 1
    total_loss /= count
    print('[Epoch: %d]\tloss: %.5f' % (epoch + 1, total_loss))                 


def validate(test_loader, model):
    model.eval()
    correct = 0
    total = 0
    count = 0
    acc = 0
    for i_batch, sample_batched in enumerate(test_loader):
        input_image = torch.autograd.Variable(sample_batched['image'].cuda())
        target_var = torch.autograd.Variable(sample_batched['y'].cuda()).float()
        model_out = model(input_image)
        
        pred = model_out.view(-1) >= 0.5
        truth = target_var >= 0.5
        acc += pred.eq(truth).sum().item() * 100.0 / target_var.size()[0]
        count += 1
            
    print('Testset accuracy: %.2f' % (acc / count))


def remove_environment(model, url):
    transform = transforms.Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    file_list = glob.glob('{}/*/*.png'.format(url))
    count = 0
    for file in file_list:
        image = io.imread(file) / 255.0
        image = transform(np.stack([image]))
        output = model(torch.from_numpy(image).cuda())
        environment = output.item()
        if environment >= 0.5:
            print(file)
            os.remove(file)
            count += 1

    print(count)
    return
    
    
def save_checkpoint(state, dirpath, epoch):
    filename = 'removal_checkpoint.{}.ckpt'.format(epoch)
    checkpoint_path = os.path.join(dirpath, filename)
    torch.save(state, checkpoint_path)


if __name__ == '__main__':
    args = data_pruning_parser()
    main(args)
