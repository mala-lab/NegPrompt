import argparse
import sys
import csv
import datetime
import importlib
import os
import time

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import wandb
import open_clip
import torchvision
from torchvision import datasets, transforms

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from core import test_clip, test_nega_clip, train_clip, train_nega_clip
from datasets.classname import *
from datasets.osr_dataloader import (
    CIFAR100_OSR,
    CIFAR10_OSR,
    MNIST_OSR,
    SVHN_OSR,
    Tiny_ImageNet_OSR,
    ImageNet1K_OSR,
    ImageNet_OOD
)
from models import scheduler_builder
from models.models import NegaPromptCLIP, OriginalCLIP
from utils import Logger, load_networks, save_networks, get_class_prototypes, train_tsne_plot_with_proto


from tqdm import tqdm
import numpy as np
from scipy import interpolate
from sklearn import metrics
from sklearn.metrics import accuracy_score as Acc
from sklearn.metrics import roc_auc_score as Auc
from sklearn.metrics import roc_curve as Roc


_tokenizer = _Tokenizer()

parser = argparse.ArgumentParser("Training")
# Distribute
parser.add_argument("--local_rank", type=int)

# Dataset
parser.add_argument('--dataset', type=str, default='mnist', help="mnist | svhn | cifar10 | cifar100 | tiny_imagenet | ImageNet_p[1-10]| OOD_ImageNet_[SUN|iNaturalist|places365|dtd")
parser.add_argument('--dataroot', type=str, default='./data')
parser.add_argument('--outf', type=str, default='./log')
parser.add_argument('--out-num', type=int, default=50, help='For CIFAR100')
# add a argument descriping the metadata


# optimization
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.1, help="learning rate for model")
parser.add_argument('--gan_lr', type=float, default=0.0002, help="learning rate for gan")
parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--stepsize', type=int, default=30)
parser.add_argument('--temp', type=float, default=1.0, help="temp")
parser.add_argument('--num-centers', type=int, default=1)

# model
parser.add_argument('--weight-pl', type=float, default=0.1, help="weight for center loss")
parser.add_argument('--beta', type=float, default=0.1, help="weight for entropy loss")
parser.add_argument('--model', type=str, default='classifier32')

# misc
parser.add_argument('--nz', type=int, default=100)
parser.add_argument('--ns', type=int, default=1)
parser.add_argument('--eval-freq', type=int, default=1)
parser.add_argument('--print-freq', type=int, default=100)
parser.add_argument('--gpu', type=str, default= '0')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--save-dir', type=str, default='../log')
parser.add_argument('--loss', type=str, default='Softmax')
parser.add_argument('--eval', action='store_true', help="Eval", default=False)
parser.add_argument('--cs', action='store_true', help="Confusing Sample", default=False)

# clip
parser.add_argument('--clip_backbone', type=str, default='ViT-B/16') #RN50 RN101 RN50x4 RN50x16 RN50x64 ViT-B/32 ViT-B/16 ViT-B/14 ViT-L/14@336px
parser.add_argument('--CSC', type=int, default=0)
parser.add_argument('--LOG', type=int, default=0) 
parser.add_argument('--stage', type=int, default=1)
parser.add_argument('--positive_pth', type=str, default='') # xxxx.pth
parser.add_argument('--negative_pth', type=str, default='') # xxxx.pth
#adversarial

parser.add_argument('--NEGA_CTX', type=int, default=1) 
parser.add_argument('--distance_weight', type=float, default=0.0001) 
parser.add_argument('--negative_weight', type=float, default=1) 
parser.add_argument('--random_negative', type=float, default=1e-2) 
parser.add_argument('--nega_nega_weight', type=float, default=0.00001) 
parser.add_argument('--open_score', type=str, default='msp')  #msp posi_nega posi_radius
parser.add_argument('--open_set_method', type=str, default='MSP') # MSP OE Wasserstein Fence
parser.add_argument('--positive_prompt', type=str, default='Positive')  # X X X X Positive Cats
parser.add_argument('--negative_prompt', type=str, default='Negative')  # X X X X Negative Cats

#Fence
parser.add_argument('--fence_alpha', type=float, default=0.5) #0-1

#Prototype
parser.add_argument('--prototype_weight', type=float, default=0)
parser.add_argument('--ori_dataset', type=str, default='ImageNet_p4')  # X X X X Negative Cats

#POMP
parser.add_argument('--POMP', type=int, default=0)
parser.add_argument('--POMP_k', type=int, default=128)

# fewshot
parser.add_argument('--few_shot', type=int, default=0) # 0:all_shot, n: n-shot


def config_options(options):
    if options['few_shot'] > 0:
        print('use few shot !')
        options['outf'] = './log/fewshot/{}'.format(options['few_shot'])
    return options

def main_worker(options):
    options = config_options(options)
    run = wandb.init(project="prompt_clip_openset", dir='.', reinit=True)
    run.config.update(options, allow_val_change=True)
    # run.define_metric("AUROC", summary="max")
    options = run.config
    torch.manual_seed(options['seed'])
    os.environ['CUDA_VISIBLE_DEVICES'] = options['gpu']
    use_gpu = torch.cuda.is_available()
    if options['use_cpu']: use_gpu = False

    if use_gpu:
        print("Currently using GPU: {}".format(options['gpu']))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(options['seed'])
    else:
        print("Currently using CPU")
    print('stage: ', options['stage'])
    # Dataset
    print("{} Preparation".format(options['dataset']))
    if 'mnist' in options['dataset']:
        Data = MNIST_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    elif 'cifar10' == options['dataset']:
        Data = CIFAR10_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    elif 'svhn' in options['dataset']:
        Data = SVHN_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    elif 'cifar100' in options['dataset']:
        Data = CIFAR10_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader = Data.train_loader, Data.test_loader
        out_Data = CIFAR100_OSR(known=options['unknown'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        outloader = out_Data.test_loader
    elif 'OOD' in options['dataset']:
        parts = options['dataset'].split("_") #OOD_ImageNet_SUN
        print(options['batch_size'])
        Data = ImageNet_OOD(ood_dataset=parts[2], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'], shot=options['few_shot'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    elif 'ImageNet' in options['dataset']:
        Data = ImageNet1K_OSR(datasplit = options['dataset'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'], few_shot=options['few_shot'], cfg = options)
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    else:
        Data = Tiny_ImageNet_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    
    
    
    options['num_classes'] = Data.num_classes
    if options['stage'] == 2 or options['stage'] == 3:
        options['max-epoch'] = 15
    # Model
    if 'ImageNet' in options['dataset']:
        classnames = Data.known
        options['CTX_INIT'] = 'a photo of a "{}"'

    else:
        classnames = classname_dic[options['dataset']]["classes"]
        known_class = Data.known
        # known_class.sort()
        # print('known_class', known_class)
        classnames = [classnames[i] for i in known_class]
        # print('classnames: ', classnames)
        options['CTX_INIT'] = classname_dic[options['dataset']]["templates"][0]
    test_labels = [classname.replace('_', ' ') for classname in classnames]
    # print(test_labels)
    options['classnames'] = test_labels
    options['N_CTX'] = 16
    print("CLIP backbone: {}".format(options['clip_backbone']))
    device = "cuda" if use_gpu else "cpu"
    clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
    
    # clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
    # clip_model, clip_preprocess = clip.load(options['clip_backbone'], device=device)

    
    # clip_model, _, _ = open_clip.create_model_and_transforms('RN50', pretrained='openai')
    # clip_model, _, _ = open_clip.create_model_and_transforms('RN101', pretrained='openai')
    # clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    
    options['clip_implement'] = 'open_clip'
    # options['clip_implement'] = 'original_clip'
    if(options['clip_implement'] == 'open_clip'):
        clip_model.dtype = torch.float32
        clip_model.visual.input_resolution = 224
    if use_gpu:
        clip_model = clip_model.cuda()
    # clip_model, clip_preprocess = clip.load(options['clip_backbone'], device=device)
    for params in clip_model.parameters():
        params.requires_grad_(False)
    model = NegaPromptCLIP(options, classnames, clip_model)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    
    # torch.cuda.set_device(options['local_rank'])
    # torch.distributed.init_process_group(backend='nccl')
    # model = model.cuda()
    # model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    
    model_path = os.path.join(options['outf'], 'models', options['dataset'])

    if not os.path.exists(model_path):
        os.makedirs(model_path)
        
    if options['stage'] == 2 or options['stage'] == 3:
        model_positive_path = '{}/{}'.format(model_path, 'md.pth')
        save_model = torch.load(model_positive_path)
        if 'prompt_learner.ctx_positive' in save_model.keys():
            model.get_ctx_posi(save_model['prompt_learner.ctx_positive'])
        else:
            model.get_ctx_posi(save_model['module.prompt_learner.ctx_positive'])
        print('Stage 1 model loaded!')
        # model.update_nega_features(options)
        del save_model
    
    if options['stage'] == 4:
        model_positive_path = '{}/{}'.format(model_path, 'md.pth')
        save_model = torch.load(model_positive_path)
        model.get_ctx_posi(save_model['prompt_learner.ctx_positive'])
        print('Stage 1 model loaded!')
        del save_model
        model_negative_path = '{}/{}/{}'.format(model_path, 'md.pth')
        print(model_negative_path)
        save_model = torch.load(model_negative_path)
        model.get_ctx_nega(save_model['prompt_learner.ctx_negative'])
        print('Stage 3 model loaded!')
        del save_model
        # model.update_nega_features(options)
    
    if options['stage'] == 5:
        model_negative_path = '{}/{}/{}'.format(model_path, 'md.pth')
        save_model = torch.load(model_negative_path)
        model.get_ctx_nega(save_model['prompt_learner.ctx_negative'])
        print('Stage 3 model loaded!')
        del save_model
        
   
        
    optimizer = torch.optim.SGD(model.parameters(), lr=options['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(50))
    # scheduler = scheduler_builder.ConstantWarmupSchedulr(
    #             optimizer, scheduler_1, 1, 1e-5)
    options.update(
        {
            'use_gpu':  use_gpu
        }
    )
    Loss = importlib.import_module('loss.'+options['loss'])
    criterion = getattr(Loss, options['loss'])(**options)
    # file_name = '{}_{}_{}_{}_{}'.format(options['clip_backbone'].replace('/',''), options['NEGA_CTX'], options['CSC'], options['open_score'], datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    file_name = 'md.pth'

    expr_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + "-" + run.name 
    start_time = time.time()
    best_acc = -1
    best_auroc = -1
    proto = 0
    # calculate the prototype loss
    if options['prototype_weight'] != 0:
        print('Get prototypes....')
        prototypes = get_class_prototypes(model, trainloader, options['num_classes'])
        proto = torch.zeros((options['num_classes'], prototypes[0].shape[0])).cuda()
        for i in range(options['num_classes']):
            proto[i] = prototypes[i]
        train_tsne_plot_with_proto(trainloader, testloader, model, proto, options['outf'], expr_name)
    results = test_nega_clip(model, criterion, testloader, outloader, epoch=0, **options)
    print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t FPR95 (%): {:.3f}\t AUPR (%): {:.3f}\t".format(
        results['ACC'], results['AUROC'], results['OSCR'], results['FPR95'], results['AUPR']
        ))
    if options['stage'] == 4 or options['stage'] == 5:
        print('Source dataset : ', options['ori_dataset'])
        print('Target dataset : ', options['dataset'])
        print('Source log : ', options['negative_pth'])
        print('Target log : ', options['positive_pth'])
        run.log(results, step = 0)
        run.finish()
        return results
    for epoch in range(options['max_epoch']):
        last_loss = 9999999999
        print("==> Epoch {}/{}".format(epoch+1, options['max_epoch']))
        this_loss = train_nega_clip(model, optimizer, scheduler, trainloader, run, epoch=epoch, proto = proto, **options)
        this_loss = round(this_loss, 8)
        print('this : ', this_loss)
        if this_loss == last_loss:
            print('the same')
            break
        last_loss = this_loss
        if options['eval_freq'] > 0 and (epoch+1) % options['eval_freq'] == 0 or (epoch+1) == options['max_epoch']:
            print("==> Test", options['loss'])
            results = test_nega_clip(model, criterion, testloader, outloader, epoch=0, **options)
            print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t FPR95 (%): {:.3f}\t AUPR (%): {:.3f}\t".format(
                 results['ACC'], results['AUROC'], results['OSCR'], results['FPR95'], results['AUPR']))
            # results = test_clip(model, criterion, testloader, outloader, epoch=0, **options)
            # print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'], results['OSCR']))
            run.log(results, step = epoch)
            if results['ACC'] > best_acc and options['LOG'] and options['stage'] == 1:
                best_acc = results['ACC']
                save_networks(model, model_path, file_name)
            if results['AUROC'] > best_auroc and options['LOG'] and options['stage'] == 3:
                best_auroc = results['AUROC']
                print("save:", model_path)
                save_networks(model, model_path, file_name)
            if results['AUROC'] > best_auroc:
                best_auroc = results['AUROC']
            run.log({'best_auroc': best_auroc}, step = epoch)
        if options['stepsize'] > 0: scheduler.step()
        # draw the t-sne plot of all the text features
        if 'ImageNet' not in options['dataset']:
            model.draw_tsne_plot(testloader, outloader, options['outf'], expr_name, epoch)  
        print('Now running stage_{}, dataset_{}, best_auroc: {}'.format(options['stage'], options['dataset'], best_auroc))
        if options['stage'] == 4:
            print('Original dataset : ', options['ori_dataset'])
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))
    run.finish()
    return results

if __name__ == '__main__':
    args = parser.parse_args()
    options = vars(args)
    img_size = 224
    results = dict()
    
    from split import splits_2020 as splits
    if 'ImageNet'in options['dataset']:
        options['img_size'] = 224
        dir_name = '{}_{}'.format(options['model'], options['loss'])
        dir_path = os.path.join(options['outf'], 'results', dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        file_name = options['dataset'] + '.csv'
        res = main_worker(options)
        sys.exit(0)
    for i in range(len(splits[options['dataset']])):
        options['dataroot'] = os.path.join(options['dataroot'], options['dataset'])
        known = splits[options['dataset']][len(splits[options['dataset']])-i-1]
        if options['dataset'] == 'cifar100':
            unknown = splits[options['dataset']+'-'+str(options['out_num'])][len(splits[options['dataset']])-i-1]
        elif options['dataset'] == 'tiny_imagenet':
            img_size = 224
            options['lr'] = 0.001
            unknown = list(set(list(range(0, 200))) - set(known))
        else:
            unknown = list(set(list(range(0, 10))) - set(known))

        options.update(
            {
                'item':     i,
                'known':    known,
                'unknown':  unknown,
                'img_size': img_size
            }
        )

        dir_name = '{}_{}'.format(options['model'], options['loss'])
        dir_path = os.path.join(options['outf'], 'results', dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        if options['dataset'] == 'cifar100':
            file_name = '{}_{}.csv'.format(options['dataset'], options['out_num'])
        else:
            file_name = options['dataset'] + '.csv'

        res = main_worker(options)
        res['unknown'] = unknown
        res['known'] = known
        results[str(i)] = res
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(dir_path, file_name))
