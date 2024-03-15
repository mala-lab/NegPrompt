import os
import sys
import errno
import random
import os.path as osp
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class AverageMeter(object):
    """Computes and stores the average and current value.
       
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
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

class Logger(object):
    """
    Write console output to external text file.
    
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

def save_networks(networks, result_dir, name=''):
    mkdir_if_missing(result_dir)
    weights = networks.state_dict()
    filename = '{}/{}.pth'.format(result_dir, name)
    torch.save(weights, filename)

def save_GAN(netG, netD, result_dir, name=''):
    mkdir_if_missing(osp.join(result_dir, 'checkpoints'))
    weights = netG.state_dict()
    filename = '{}/{}_G.pth'.format(result_dir, name)
    torch.save(weights, filename)
    weights = netD.state_dict()
    filename = '{}/{}_D.pth'.format(result_dir, name)
    torch.save(weights, filename)

def load_networks(networks, result_dir, name='', loss='', criterion=None):
    weights = networks.state_dict()
    filename = '{}/checkpoints/{}_{}.pth'.format(result_dir, name, loss)
    networks.load_state_dict(torch.load(filename))
    if criterion:
        weights = criterion.state_dict()
        filename = '{}/checkpoints/{}_{}_criterion.pth'.format(result_dir, name, loss)
        criterion.load_state_dict(torch.load(filename))

    return networks, criterion

def get_class_prototypes(model, trainloader, nclass):
    all_prototype = {}
    feature_num = {}
    for batch_idx, (data, labels) in enumerate(trainloader):
        data, labels = data.cuda(), labels.cuda()
        image_features = model.get_visual_features(data)
        for i in range(image_features.shape[0]):
            label = int(labels[i].item())
            feature = image_features[i,:] 
            if label not in all_prototype.keys():
                all_prototype[label] = feature
                feature_num[label] = 1
            else:
                all_prototype[label] += feature
                feature_num[label] += 1
    for label in all_prototype:
        all_prototype[label] /= feature_num[label]
    return all_prototype

def train_tsne_plot_with_proto(trainloader, testloader, model, proto, log_dir, expr_name):
    n_class = proto.shape[0]
    proto = proto.cuda()
    all_image_feature = torch.Tensor().cuda()
    all_image_label = torch.Tensor()
    for data, labels in trainloader:
        data, labels = data.cuda(), labels.cuda()
        with torch.set_grad_enabled(False):
            image_features = model.get_visual_features(data)
            all_image_feature = torch.cat([all_image_feature, image_features], dim=0)
            all_image_label = torch.cat([all_image_label, labels.cpu()], dim=0)
    for data, labels in testloader:
        data, labels = data.cuda(), labels.cuda()
        labels = -1 * (labels + 1)
        with torch.set_grad_enabled(False):
            image_features = model.get_visual_features(data)
            all_image_feature = torch.cat([all_image_feature, image_features], dim=0)
            all_image_label = torch.cat([all_image_label, labels.cpu()], dim=0)
    # total_feature = torch.cat([all_image_feature, proto], dim=0)
    # total_label = torch.cat([all_image_label, -1 * torch.Tensor(range(n_class))-1], dim=0)
    total_feature = all_image_feature
    total_label = all_image_label
    X = total_feature.cpu().detach().numpy()
    tsne_model = TSNE(metric="precomputed", n_components=2, init="random", perplexity=30)
    distance_matrix = pairwise_distances(X, X, metric='cosine', n_jobs=-1)

    data = torch.Tensor(tsne_model.fit_transform(distance_matrix))
    target = total_label
    
    dataset = TensorDataset(data, target)
    loader = DataLoader(dataset, batch_size=256)
    plt.figure()
    for x, y in loader:
        idx_real_image = (y >= 0)
        idx_test_image = (y < 0)
        # idx_proto_image = (y < 0)
        plt.scatter(x[idx_real_image, 0], x[idx_real_image, 1], marker = 'x',c = y[idx_real_image], alpha=0.2,
                        cmap=plt.cm.get_cmap("rainbow", n_class + 1), label='real')
        plt.scatter(x[idx_test_image, 0], x[idx_test_image, 1], marker = 's',c = -1 * y[idx_test_image] - 1, alpha=0.2,
                        cmap=plt.cm.get_cmap("rainbow", n_class + 1), label='real')
        # plt.scatter(x[idx_proto_image, 0], x[idx_proto_image, 1], marker = 'o',c= -1 * y[idx_proto_image] - 1, 
        #             cmap=plt.cm.get_cmap("summer", n_class + 1), label='proto')
    dir_path = os.path.join(log_dir, 'tsne', expr_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)   
    plt.savefig(os.path.join(dir_path, 'tsne_plot_proto.png'))
    plt.close()
    
def label_transform(input_list, K, N):
    """
    Pick K-len(input_list) random numbers excluding the ones in the input list from 0 to N, 
    so that there is no repetition with the numbers in the input list.
    Then, map the numbers in the list and the random numbers to integers between 0 and K,
    and return the mapping as a dictionary.
    """
    nums = set(input_list)
    while len(nums) < K:
        new_num = random.randint(0, N)
        if new_num not in nums:
            nums.add(new_num)
    ori_to_modify = {}
    modify_to_ori = {}

    map_counter = 0
    for num in sorted(nums):
        ori_to_modify[num] = map_counter
        modify_to_ori[map_counter] = num
        map_counter += 1
    return ori_to_modify, modify_to_ori  