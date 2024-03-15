import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from utils import AverageMeter, label_transform
from geomloss import SamplesLoss

def custom_alpha_cross_entropy(predict, soft_label, alpha):
    soft_label = soft_label.bool()
    softmax_p = F.softmax(predict, dim=1)
    sub = torch.masked_select(softmax_p, soft_label).view(predict.shape[0], -1)
    sub = sub[1:,:]
    # predict_class_prob = softmax_p.gather(1, label.view(-1, 1)).squeeze()
    diff = torch.abs(sub - alpha/(1+sub.shape[1]))
    loss = -torch.mean(diff)
    return loss
import random


def custom_cost(X,Y):
    if len(X.shape) == 2:
        N, D = X.shape
        M, D = Y.shape
        return (1-torch.eye(N,M)).cuda()
    if len(X.shape) == 3:
        B, N, D = X.shape
        B, M, D = Y.shape
        return torch.unsqueeze(1 - torch.eye(N, M), 0).repeat(B,1,1).cuda()
        
def train_clip(net, optimizer, scheduler, trainloader, run, epoch=None,   **options):
    losses = AverageMeter()
    loss_all = 0
    ori_to_modify, modify_to_ori = None, None
    for batch_idx, (data, labels) in enumerate(trainloader):
        if options['use_gpu']:
            data, labels = data.cuda(), labels.cuda()
        
        with torch.set_grad_enabled(True):
            output, _  = net(data)
            loss = F.cross_entropy(output, labels)
            net.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        losses.update(loss.item(), labels.size(0))
        run.log({'loss': loss.item()})
        if (batch_idx+1) % options['print_freq'] == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})" \
                  .format(batch_idx+1, len(trainloader), losses.val, losses.avg)) 
        loss_all += losses.avg

    return loss_all

def train_nega_clip(net, optimizer, scheduler, trainloader, run, epoch=None,  proto=None, **options):
    losses = AverageMeter()
    loss_all = 0
    n_nega_ctx = options['NEGA_CTX']
    for batch_idx, (data, labels) in enumerate(trainloader):
        if options['use_gpu']:
            data, labels = data.cuda(), labels.cuda()
        
        if options['POMP']:
            ori_to_modify, modify_to_ori = label_transform(labels.cpu().numpy(), options['POMP_k'], options['num_classes']-1)
            modified_labels = torch.tensor([ori_to_modify[label.item()] for label in labels]).cuda()
            labels = modified_labels
        else:
            ori_to_modify, modify_to_ori = None, None
             
        with torch.set_grad_enabled(True):
            output, text_features = net(data, modify_to_ori)
            # output.shape = [batch_size, nclass * 1+n_nega_ctx]
            output_posi = output.view(-1, int(output.shape[1]/(1+n_nega_ctx)), 1+n_nega_ctx)[:, :, 0]
            ensemble_text_features = text_features.view(int(text_features.shape[0]/(1+n_nega_ctx)), 1+n_nega_ctx, -1)
            positive_text_features = ensemble_text_features[:, 0, :]
            negative_text_features = ensemble_text_features[:, 1:, :] # (batch_size, n_nega_ctx , 512)
    
            loss_positive = F.cross_entropy(output_posi, labels)
            loss_prototype = 0
            if(options['prototype_weight'] != 0):
                loss_prototype = -torch.sum(torch.mul(positive_text_features, proto))
                
            loss_nega_to_other = 0
            loss_nega_to_posi = 0
            loss_nega_to_nega = 0
            
            if options['stage'] > 1:
                loss_positive *= 1e-8
                # negative_features = negative_features.view(0)
                for i in range(negative_text_features.shape[0]):
                    negative_features = negative_text_features[i,:,:].float()
                    negative_features_mean = torch.mean(negative_features, dim=0, keepdim=True)
                    negative_features_mean_norm = negative_features_mean.norm(dim=-1, keepdim=True)

                    # Euclidean distance
                    # loss_nega_to_nega += -sum(torch.pdist(negative_features, p=2))

                    # Cosine distance
                    negative_features_norm = negative_features.norm(dim=-1, keepdim=True)
                    # nega_nega
                    # dot_product = negative_features_norm @ negative_features_norm.t()
                    # nega_mean
                    dot_product = negative_features_norm @ negative_features_mean_norm.t()
                    loss_nega_to_nega += -torch.mean(1-dot_product)
                loss_nega_to_nega /= negative_text_features.shape[0]   
                
                # print(output_negas.transpose(1,2))
                out_nega_forCE = output
                # create soft_target(1-hot) for negative samples and positive samples
                soft_target = torch.zeros(out_nega_forCE.shape).long().cuda()
                idx = torch.arange(out_nega_forCE.shape[0]).cuda()
                soft_target.view(soft_target.shape[0], int(output.shape[1]/(1+n_nega_ctx)), -1)[idx, labels, :] = 1
                labels_nega = labels.reshape(1, -1).repeat(n_nega_ctx, 1).t().reshape(-1)
                if options['open_set_method'] == 'MSP':
                    loss_fun = nn.MultiLabelSoftMarginLoss(reduction='mean')
                    loss_nega_to_other = loss_fun(out_nega_forCE, soft_target)
                    # loss_nega_to_other = F.cross_entropy(out_nega_forCE, labels_nega)
                elif options['open_set_method'] == 'Fence':
                    loss_nega_to_other = custom_alpha_cross_entropy(out_nega_forCE, soft_target, alpha=options['fence_alpha'])
                elif options['open_set_method'] == 'OE':
                    loss_nega_to_other = -(out_nega_forCE.mean(1) - torch.logsumexp(out_nega_forCE, dim=1)).mean() #OE
                # elif options['open_set_method'] == 'Wasserstein':
                #     labels_openset = torch.eye(output_negas.shape[1], output_negas.shape[1]).cuda()
                #     labels_openset = labels_openset.unsqueeze(-1)
                #     #softmax out_nega_forCE
                #     sm_out = F.softmax(out_nega_forCE, dim=1).cuda()
                #     wass_nega = sm_out.unsqueeze(-1)
                #     wood_loss = SamplesLoss(loss="sinkhorn", diameter=1., p=2, blur=1., cost = custom_cost)
                #     batch_size = wass_nega.shape[0]
                #     wass_loss = torch.zeros(batch_size, output_negas.shape[1]).cuda()
                #     for b in range(batch_size):
                #         input_b = wass_nega[b:b+1, :, :].repeat(output_negas.shape[1], 1, 1).float().cuda()
                #         wass_loss[b] = wood_loss(input_b[:,:,0], input_b, labels_openset[:,:,0], labels_openset)
                #     values, idx = torch.min(wass_loss, dim=1)
                #     loss_nega_to_other = -torch.mean(values)
                else:
                    raise NotImplementedError
                all_class_dis = 0
                for i in range(negative_text_features.shape[0]):
                    positive_feature = positive_text_features[i:i+1,:].float()
                    negative_feature = negative_text_features[i,:,:].float()
                    positive_feature_norm = positive_feature/positive_feature.norm(dim=-1, keepdim=True)
                    negative_feature_norm = negative_feature/negative_feature.norm(dim=-1, keepdim=True)
                    dot_product = positive_feature_norm @ negative_feature_norm.t()
                    mean_cosine_dis = (1-dot_product).mean()
                    all_class_dis += mean_cosine_dis
                    
                if options['open_set_method'] == 'MSP':
                    loss_nega_to_posi -= all_class_dis/negative_text_features.shape[0]
                elif options['open_set_method'] == 'Fence':
                    loss_nega_to_posi = 0
                else:
                    loss_nega_to_posi += all_class_dis/negative_text_features.shape[0]
                
            loss = loss_positive + options['prototype_weight'] * loss_prototype \
                    + options['negative_weight']*loss_nega_to_other + options['distance_weight']*loss_nega_to_posi + options['nega_nega_weight']*loss_nega_to_nega

            net.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        losses.update(loss.item(), labels.size(0))
        
        if (batch_idx+1) % options['print_freq'] == 0: 
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})" \
                  .format(batch_idx+1, len(trainloader), losses.val, losses.avg))
        
        loss_all += losses.avg
    run.log({'loss': loss_all}, step=epoch)
    return loss_all

