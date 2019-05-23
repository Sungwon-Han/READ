import os
import csv
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataloader import *
from parameters import *

def main(args):   
    LOAD = args.load
    LOAD_PATH = args.modelurl
    
    train_gps = ProxyDataset(metadata = "./data/proxy/proxy_metadata_train.csv", root_dir = "./data/proxy/train/",
                         transform=transforms.Compose([
                             RandomRotate(),
                             ToTensor(),
                             Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                         ]))

    test_gps = ProxyDataset(metadata = "./data/proxy/proxy_metadata_test.csv", root_dir = "./data/proxy/test/",
                            transform=transforms.Compose([
                                 ToTensor(),
                                 Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ]))

    train_loader = torch.utils.data.DataLoader(train_gps, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(test_gps, batch_size=len(test_gps), shuffle=False, num_workers=args.workers)
 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained = True)   
    model.fc = nn.Sequential(nn.Linear(512, 3), nn.Softmax())
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    if LOAD == False:           
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
        for epoch in range(args.epochs):
            train(train_loader, model, optimizer, epoch, args.batch_size)
            if (epoch + 1) % args.evaluation_epochs == 0:
                validate(test_loader, model)

            if (epoch + 1) % args.checkpoint_epochs == 0:
                save_checkpoint({'state_dict': model.state_dict()}, "./model", model, epoch + 1)
    else:
        model.load_state_dict(torch.load(LOAD_PATH)['state_dict'], strict=False)
        model.to(device)
        print("Load Finished")
    
    if torch.cuda.device_count() > 1:
        model.module.fc = nn.Sequential()
    else:
        model.fc = nn.Sequential()
        
    model.eval()
    extract_features(model)     
                
                
def train(train_loader, model, optimizer, epoch, batch_size):
    model.train()
    total_loss = 0
    count = 0
    for i_batch, sample_batched in enumerate(train_loader):
        input_image = torch.autograd.Variable(sample_batched['image'].cuda())
        t = torch.autograd.Variable(sample_batched['y'].cuda())
        y = model(input_image)     
        loss = torch.mean(torch.sum(- t * torch.log(y), 1))
        
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
        target_var = torch.autograd.Variable(sample_batched['y'].cuda())
        model_out = model(input_image)
        
        _, predicted = torch.max(model_out.data, 1)
        _, answer =  torch.max(target_var.data, 1)
        
        total += input_image.size(0)
        correct += (predicted == answer).sum().item()
            
    print('Testset accuracy: %.2f' % (correct / total * 100.0))

    
def save_checkpoint(state, dirpath, model, epoch):
    filename = 'proxy_checkpoint.{}.ckpt'.format(epoch)
    checkpoint_path = os.path.join(dirpath, filename)
    torch.save(state, checkpoint_path)

                
def extract_features(model):
    train_district = GPSDataset(metadata = "./data/sample_train/metadata.csv", root_dir = "./data/sample_train/original/",
                                transform=transforms.Compose([
                                   ToTensor(),
                                   Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ]))

    test_district = GPSDataset(metadata = "./data/sample_test/metadata.csv", root_dir = "./data/sample_test/original/",
                               transform=transforms.Compose([
                                  ToTensor(),
                                  Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                               ]))

    train_loader = torch.utils.data.DataLoader(train_district, batch_size=1, shuffle=False, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_district, batch_size=1, shuffle=False, num_workers=1)
    
    count = 0
    for batch in enumerate(train_loader):
        print("Reducing Train data - count : {}".format(count))
        input_images = torch.autograd.Variable(batch[1]['images'][0].cuda(async=True))
        folder_idx = batch[1]['folder_idx'].item()
        model_out = model(input_images)
        np.savetxt("./data/sample_train/reduced/{}.csv".format(folder_idx), model_out.cpu().detach().numpy())
        count += 1
    print("train_reduce finished")
    
    count = 0    
    for batch in enumerate(test_loader):
        print("Reducing Test data - count : {}".format(count))
        input_images = torch.autograd.Variable(batch[1]['images'][0].cuda(async=True))
        folder_idx = batch[1]['folder_idx'].item()        
        model_out = model(input_images)          
        np.savetxt("./data/sample_test/reduced/{}.csv".format(folder_idx), model_out.cpu().detach().numpy())
        count += 1
    print("test_reduce finished") 

if __name__ == '__main__':
    args = extract_embeddings_parser()
    main(args)
