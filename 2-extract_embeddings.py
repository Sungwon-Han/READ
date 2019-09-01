import os
import csv
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as Func
import torchvision.models as models
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse

import numpy as np
from dataloader import *
from parameters import *
from math import log

def main(args):    
    global global_step
    global_step = 0
    train_labeled = ProxyDataset(metadata = "./data/proxy/proxy_metadata_train.csv", 
                                 root_dir = "./data/proxy/train/",
                                 transform=transforms.Compose([
                                     RandomRotate(),
                                     ToTensor(),
                                     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                 ]))
    
    train_unlabeled = UnlabeledDataset(root_dir = "./data/unlabeled",
                                       transform=transforms.Compose([
                                          transforms.RandomVerticalFlip(p=0.5),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                       ]))

    test_gps = ProxyDataset(metadata = "./data/proxy/proxy_metadata_test.csv", 
                            root_dir = "./data/proxy/test/",
                            transform=transforms.Compose([
                                 ToTensor(),
                                 Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ]))

    train_labeled_loader = torch.utils.data.DataLoader(train_labeled, batch_size=args.batch_size, shuffle=True, num_workers=4)
    train_unlabeled_loader = torch.utils.data.DataLoader(train_unlabeled, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    test_loader = torch.utils.data.DataLoader(test_gps, batch_size=20, shuffle=False, num_workers=4)
 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained = True)   
    model.fc = nn.Sequential(nn.Linear(512, 3), nn.Softmax())
    ema_model = models.resnet18(pretrained = False)   
    ema_model.fc = nn.Sequential(nn.Linear(512, 3), nn.Softmax())
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    if args.load == False:           
        model.to(device)
        ema_model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
        for epoch in range(args.epochs):
            train(train_labeled_loader, train_unlabeled_loader, model, ema_model, optimizer, epoch, args.batch_size)
            if (epoch + 1) % args.evaluation_epochs == 0:
                validate(test_loader, model)

            if (epoch + 1) % args.checkpoint_epochs == 0:
                save_checkpoint({'state_dict': model.state_dict()}, "./model", model, epoch + 1)
    else:
        model.load_state_dict(torch.load(args.modelurl)['state_dict'], strict=False)
        model.to(device)
        print("Load Finished")
    
    if torch.cuda.device_count() > 1:
        model.module.fc = nn.Sequential()
    else:
        model.fc = nn.Sequential()
        
    model.eval()
    extract_features(model)     
                
                
def softmax_mse_loss(input_logits, target_logits):
    num_classes = input_logits.size()[1]
    return Func.mse_loss(input_logits, target_logits, reduction='sum') / num_classes            
   
    
def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))
     
        
def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)            
                
def train(train_labeled_loader, train_unlabeled_loader, model, ema_model, optimizer, epoch, batch_size):
    global global_step
    model.train()
    ema_model.eval()
    total_loss = 0
    total_supervised_loss = 0
    total_unsupervised_loss = 0
    count = 0
    consistency_criterion = softmax_mse_loss    
    
    unlabeled_iter = iter(train_unlabeled_loader)
    for i_batch, sample_batched in enumerate(train_labeled_loader):   
        ema_input = next(unlabeled_iter)
        ema_input_var = torch.autograd.Variable(ema_input).cuda()
        input_image = torch.autograd.Variable(sample_batched['image'].cuda())
        total_input = torch.cat((input_image, ema_input_var), dim = 0)
 
        t = torch.autograd.Variable(sample_batched['y'].cuda())
        y_total = model(total_input)
        y = y_total[:batch_size]
        y_ema = y_total[batch_size:]
        
        supervised_loss = torch.mean(torch.sum(- t * torch.log(y), 1))
       
        ema_model_out = ema_model(ema_input_var)
        ema_logit = ema_model_out
        ema_logit = Variable(ema_logit.detach().data, requires_grad=False)
        consistency_weight = get_current_consistency_weight(epoch)
        consistency_loss = consistency_weight * consistency_criterion(y_ema, ema_logit) / (batch_size * 5)        

        loss = supervised_loss + consistency_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_supervised_loss += supervised_loss.item()
        total_unsupervised_loss += consistency_loss.item()
        count += 1
        global_step += 1
        update_ema_variables(model, ema_model, args.ema_decay, global_step)        
        
    total_loss /= count
    total_supervised_loss /= count
    total_unsupervised_loss /= count
    print(consistency_weight)
    print('[Epoch: %d]\tsuploss: %.5f\tunsuploss: %.5f\tloss: %.5f' % (epoch + 1, total_supervised_loss, total_unsupervised_loss, total_loss))     
       
    
def validate(test_loader, model):
    model.eval()
    correct = 0
    total = 0
    count = 0
    acc = 0
    loss = 0
    batch_num = 0
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(test_loader):
            input_image = torch.autograd.Variable(sample_batched['image'].cuda())
            target_var = torch.autograd.Variable(sample_batched['y'].cuda())
            model_out = model(input_image)
            supervised_loss = torch.mean(torch.sum(- target_var * torch.log(model_out), 1))

            _, predicted = torch.max(model_out.data, 1)
            _, answer =  torch.max(target_var.data, 1)

            total += input_image.size(0)
            correct += (predicted == answer).sum().item()
            loss += supervised_loss
            batch_num += 1

        print('Testset accuracy: %.2f' % (correct / total * 100.0))
        print('Test loss: %.5f' % (loss / batch_num))
    return loss / batch_num
 

def extract_features(model):
    train_district = GPSDataset(metadata = "./data/train/metadata_train.csv", 
                                root_dir = "./data/train/",
                                transform=transforms.Compose([
                                   ToTensor(),
                                   Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ]))

    test_district = GPSDataset(metadata = "./data/test/metadata_test.csv", 
                               root_dir = "./data/test/",
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
        np.savetxt("./data/train/reduced/{}.csv".format(folder_idx), model_out.cpu().detach().numpy())
        count += 1
    print("train_reduce finished")
    
    count = 0    
    for batch in enumerate(test_loader):
        print("Reducing Test data - count : {}".format(count))
        input_images = torch.autograd.Variable(batch[1]['images'][0].cuda(async=True))
        folder_idx = batch[1]['folder_idx'].item()        
        model_out = model(input_images)          
        np.savetxt("./data/test/reduced/{}.csv".format(folder_idx), model_out.cpu().detach().numpy())
        count += 1
    print("test_reduce finished")     
    

def get_current_consistency_weight(epoch):
    return args.consistency * sigmoid_rampup(epoch, args.rampup)    
    
def save_checkpoint(state, dirpath, model, epoch, arch_name = "resnet18"):
    filename = '{}_{}.ckpt'.format(arch_name, epoch)
    checkpoint_path = os.path.join(dirpath, filename)
    torch.save(state, checkpoint_path)

if __name__ == '__main__':
    args = extract_embeddings_parser()
    main(args)
