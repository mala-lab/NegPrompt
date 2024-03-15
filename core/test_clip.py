import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import sklearn.metrics as sk
from utils import label_transform
from core import evaluation
from transformers import CLIPTokenizer
from transformers import CLIPModel
import time

def test_clip(net, criterion, testloader, outloader, epoch=None, **options):

    
    
    correct, total = 0, 0

    torch.cuda.empty_cache()

    _pred_k, _pred_u, _labels = [], [], []
    known_number = {}
    correct_number = {}
    all_results = {}
    with torch.no_grad():
        for data, labels in testloader:
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            
            
            with torch.set_grad_enabled(False):
                logits,_ = net(data)
                predictions = logits.data.max(1)[1]
                total += labels.size(0)
                correct += (predictions == labels.data).sum()
                for i in range(len(labels.data)):
                    if labels.data[i].item() not in known_number.keys():
                        known_number[labels.data[i].item()] = 0
                        correct_number[labels.data[i].item()] = 0
                        all_results[labels.data[i].item()] = {}
                    if predictions[i].item() not in all_results[labels.data[i].item()].keys():
                        all_results[labels.data[i].item()][predictions[i].item()] = 0
                    all_results[labels.data[i].item()][predictions[i].item()] += 1
                    known_number[labels.data[i].item()] += 1
                    if predictions[i] == labels.data[i]:
                        correct_number[labels.data[i].item()] += 1
                                        
                _pred_k.append(logits.data.cpu().numpy())
                _labels.append(labels.data.cpu().numpy())

        for batch_idx, (data, labels) in enumerate(outloader):
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            
            with torch.set_grad_enabled(False):
                logits,_ = net(data)
                ood_score = logits.data.cpu().numpy()
    # # class_acc
    # class_acc = {}
    # for key in known_number.keys():
    #     class_acc[key] = correct_number[key] / known_number[key]
    # print('class_acc: ', class_acc  )
    # # print all_result
    # for key in all_results.keys():
    #     print('class ', key)
    #     print(all_results[key])
    # Accuracy
    acc = float(correct) * 100. / float(total)
    print('Acc: {:.5f}'.format(acc))

    _pred_k = np.concatenate(_pred_k, 0)
    _pred_u = np.concatenate(_pred_u, 0)
    _labels = np.concatenate(_labels, 0)
    
    # Out-of-Distribution detction evaluation
    x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
    results = evaluation.metric_ood(x1, x2)['Bas']
    
    # OSCR
    _oscr_socre = evaluation.compute_oscr(_pred_k, _pred_u, _labels)

    results['ACC'] = acc
    results['OSCR'] = _oscr_socre * 100.

    return results

def test_nega_clip(net, criterion, testloader, outloader, epoch=None, **options):
    correct, total = 0, 0
    _pred_k, _pred_u, _labels = [], [], []
    logits_posi_id, logits_nega_id, logits_posi_ood, logits_nega_ood = [], [], [], []
    net.eval()
    with torch.no_grad():
        if torch.cuda.device_count() > 1:
            prompts = net.module.prompt_learner()
            tokenized_prompts = net.module.tokenized_prompts
            text_features = net.module.text_encoder(prompts, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        else:
            prompts = net.prompt_learner()
            tokenized_prompts = net.tokenized_prompts
            text_features = net.text_encoder(prompts, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        torch.cuda.empty_cache()
        # breakpoint()
        tqdm_object = tqdm(testloader, total=len(testloader))
        for batch_idx, (data, labels) in enumerate(tqdm_object):
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            if torch.cuda.device_count() > 1:
                logits, _ = net.module.forward_test(data, text_features)
                logits /= net.module.logit_scale.exp()
            else:
                logits, _ = net.forward_test(data, text_features)
                logits /= net.logit_scale.exp()
            predictions, ood_score, logits_posi, logits_negas = get_ood_score(logits, options)
            _pred_k.append(ood_score)
            correct += (predictions == labels.data).sum()
            total += labels.size(0)
            _labels.append(labels.data.cpu().numpy())
            logits_posi_id.append(logits_posi.data.cpu().numpy())
            logits_nega_id.append(logits_negas.data.cpu().numpy())
        acc = float(correct) * 100. / float(total)
        print('Acc: {:.5f}'.format(acc))
        tqdm_object = tqdm(outloader, total=len(outloader))
        for batch_idx, (data, labels) in enumerate(tqdm_object):
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            
            with torch.set_grad_enabled(False):
                if torch.cuda.device_count() > 1:
                    logits, _ = net.module.forward_test(data, text_features)
                    logits /= net.module.logit_scale.exp()
                else:
                    logits, _ = net.forward_test(data, text_features)
                    logits /= net.logit_scale.exp()
                predictions, ood_score, logits_posi, logits_negas = get_ood_score(logits, options)
                _pred_u.append(ood_score)
                logits_posi_ood.append(logits_posi.data.cpu().numpy())
                logits_nega_ood.append(logits_negas.data.cpu().numpy())
                
    acc = float(correct) * 100. / float(total)
    print('Acc: {:.5f}'.format(acc))

    _pred_k = np.concatenate(_pred_k, 0)
    _pred_u = np.concatenate(_pred_u, 0)
    _labels = np.concatenate(_labels, 0)
    
    # Out-of-Distribution detction evaluation
    x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
    results = evaluation.metric_ood(x1, x2)['Bas']
    
    # save _pred_k, -pred_u
    # score_dic = {}
    # score_dic['pred_k'] = _pred_k
    # score_dic['pred_u'] = _pred_u
    # score_dic['logits_posi_id'] = np.concatenate(logits_posi_id, 0)
    # score_dic['logits_nega_id'] = np.concatenate(logits_nega_id, 0)
    # score_dic['logits_posi_ood'] = np.concatenate(logits_posi_ood, 0)
    # score_dic['logits_nega_ood'] = np.concatenate(logits_nega_ood, 0)
    # np.save('savescores/' + options['dataset'] + '_ score_dic.npy', score_dic)
    # OSCR
    _oscr_socre = evaluation.compute_oscr(_pred_k, _pred_u, _labels)

    auroc, aupr, fpr95 = compute_fpr(x1, x2)
    results['ACC'] = acc
    results['OSCR'] = _oscr_socre * 100.
    results['FPR95'] = fpr95 * 100.
    results['AUPR'] = aupr * 100.
    return results

def compute_fpr(pred_k, pred_u):
        x1 = pred_k
        x2 = pred_u
        pos = np.array(x1[:]).reshape((-1, 1))
        neg = np.array(x2[:]).reshape((-1, 1))
        examples = np.squeeze(np.vstack((pos, neg)))
        labels = np.zeros(len(examples), dtype=np.int32)
        labels[:len(pos)] += 1
        
        auroc = sk.roc_auc_score(labels, examples)
        aupr = sk.average_precision_score(labels, examples)
        fpr95 = fpr_and_fdr_at_recall(labels, examples)
        
        
        # fpr,tpr,thresh = roc_curve(labels, examples, pos_label=1)
        # fpr95 = float(interpolate.interp1d(tpr, fpr)(0.95))
        return auroc, aupr, fpr95
        
def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out

def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])

def get_ood_score(logits, options):
    n_nega_ctx = options['NEGA_CTX']
    softmax_logits = F.softmax(logits, dim=1)
    softmax_logits = softmax_logits.view(-1, int(softmax_logits.shape[1]/(1+n_nega_ctx)), 1+n_nega_ctx)
    logits = logits.view(-1, int(logits.shape[1]/(1+n_nega_ctx)), 1+n_nega_ctx)
    
    softmax_logits_posi = softmax_logits[:, :, 0]
    softmax_logits_negas = softmax_logits[:, :, 1:]
    logits_posi = logits[:, :, 0]
    logits_negas = logits[:, :, 1:]
    predictions = softmax_logits_posi.data.max(1)[1]

    if options['open_score'] == 'msp':
        ood_score = softmax_logits_posi.data.cpu().numpy()
    elif options['open_score'] == 'maxlogit':
        ood_score = logits_posi.data.cpu().numpy()
    elif options['open_score'] == 'energy_oe':
        energy = torch.log(torch.sum(torch.exp(logits_posi), dim=1)).unsqueeze(1).cpu().numpy()
        ood_score = energy
    elif options['open_score'] == 'nega':
        ood_score = softmax_logits_negas.data.max(2)[0].cpu().numpy()
    elif options['open_score'] == 'posi_nega':
        nega_dis = torch.Tensor(softmax_logits_posi.shape[0]).cuda()
        for i in range(softmax_logits_posi.shape[0]):
            nega_dis[i] = torch.max(softmax_logits_negas[i, predictions[i], :])
        nega_dis = nega_dis.view(-1, 1)
        nega_dis = nega_dis.repeat(1, softmax_logits_posi.shape[1])
        posi_minus_nega = softmax_logits_posi - nega_dis
        ood_score = posi_minus_nega.data.cpu().numpy()
    elif options['open_score'] == 'posi_minus_closest_radius':
        _, min_loc = torch.min(softmax_logits_negas, dim=2)
        index1 = torch.arange(min_loc.shape[1])
        index1 = index1.repeat(min_loc.shape[0]).cuda()
        index2 = min_loc.flatten().cuda()
        right_radius = radius[index1, index2].view(min_loc.shape[0], min_loc.shape[1]).cuda()
        posi_minus_radius = right_radius - softmax_logits_posi
        ood_score = posi_minus_radius.data.cpu().numpy()
    elif options['open_score'] == 'posi_radius':
        #right_radius(logits_posi.shape[0] * right_radius.shape[0]) is repeated by radius_mean\
        right_radius = radius_mean.expand((softmax_logits_posi.shape[0], -1)).cuda()
        posi_minus_radius = right_radius - softmax_logits_posi
        ood_score = posi_minus_radius.data.cpu().numpy()
    return predictions, ood_score, logits_posi, logits_negas
    


# def print_measures(log, auroc, aupr, fpr, method_name='Ours', recall_level=0.95):
#     if log == None: 
#         print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
#         print('AUROC: \t\t\t{:.2f}'.format(100 * auroc))
#         print('AUPR:  \t\t\t{:.2f}'.format(100 * aupr))
#     else:
#         log.debug('\t\t\t\t' + method_name)
#         log.debug('  FPR{:d} AUROC AUPR'.format(int(100*recall_level)))
#         log.debug('& {:.2f} & {:.2f} & {:.2f}'.format(100*fpr, 100*auroc, 100*aupr))
        
# def get_and_print_results(log, in_score, out_score):
#     '''
#     1) evaluate detection performance for a given OOD test set (loader)
#     2) print results (FPR95, AUROC, AUPR)
#     '''
#     aurocs, auprs, fprs = [], [], []
#     measures = get_measures(-in_score, -out_score)
#     aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
#     print(f'in score samples (random sampled): {in_score[:3]}, out score samples: {out_score[:3]}')
#     # print(f'in score samples (min): {in_score[-3:]}, out score samples: {out_score[-3:]}')
#     auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
#     print_measures(log, auroc, aupr, fpr)
    

# def get_measures(_pos, _neg, recall_level=0.95):
#     pos = np.array(_pos[:]).reshape((-1, 1))
#     neg = np.array(_neg[:]).reshape((-1, 1))
#     examples = np.squeeze(np.vstack((pos, neg)))
#     labels = np.zeros(len(examples), dtype=np.int32)
#     labels[:len(pos)] += 1

#     auroc = sk.roc_auc_score(labels, examples)
#     aupr = sk.average_precision_score(labels, examples)
#     fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

#     return auroc, aupr, fpr