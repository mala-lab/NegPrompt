import os
import torch
import numpy as np
import pandas as pd
import json
import math
import re
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, SVHN
from torch.utils.data import  Dataset

BICUBIC = Image.BICUBIC
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
class MNISTRGB(MNIST):
    """MNIST Dataset.
    """
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L')
        img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class MNIST_Filter(MNISTRGB):
    """MNIST Dataset.
    """
    def __Filter__(self, known):
        targets = self.targets.data.numpy()
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(known.index(targets[i]))
        self.targets = np.array(new_targets)
        mask = torch.tensor(mask).long()
        self.data = torch.index_select(self.data, 0, mask)

class MNIST_OSR(object):
    def __init__(self, known, dataroot='./data/mnist', use_gpu=True, num_workers=8, batch_size=128, img_size=32):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 10))) - set(known))

        print('Selected Labels: ', known)

        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        # train_transform = transforms.Compose([
        #     Resize(img_size, interpolation=BICUBIC),
        #     CenterCrop(img_size),
        #     ToTensor(),
        #     Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        # ])
        # transform = transforms.Compose([
        #     Resize(img_size, interpolation=BICUBIC),
        #     CenterCrop(img_size),
        #     ToTensor(),
        #     Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        # ])
        
        pin_memory = True if use_gpu else False

        trainset = MNIST_Filter(root=dataroot, train=True, download=True, transform=train_transform)
        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known)
        
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        
        testset = MNIST_Filter(root=dataroot, train=False, download=True, transform=transform)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)
        
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        outset = MNIST_Filter(root=dataroot, train=False, download=True, transform=transform)
        outset.__Filter__(known=self.unknown)

        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))

class CIFAR10_Filter(CIFAR10):
    """CIFAR10 Dataset.
    """
    def __Filter__(self, known):
        datas, targets = np.array(self.data), np.array(self.targets)
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(known.index(targets[i]))
        self.data, self.targets = np.squeeze(np.take(datas, mask, axis=0)), np.array(new_targets)


class CIFAR10_OSR(object):
    def __init__(self, known, dataroot='./data/cifar10', use_gpu=True, num_workers=8, batch_size=128, img_size=32):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 10))) - set(known))

        print('Selected Labels: ', known)

        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False

        trainset = CIFAR10_Filter(root=dataroot, train=True, download=True, transform=train_transform)
        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known)
        
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        
        testset = CIFAR10_Filter(root=dataroot, train=False, download=True, transform=transform)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)
        
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        outset = CIFAR10_Filter(root=dataroot, train=False, download=True, transform=transform)
        outset.__Filter__(known=self.unknown)

        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))

class CIFAR100_Filter(CIFAR100):
    """CIFAR100 Dataset.
    """
    def __Filter__(self, known):
        datas, targets = np.array(self.data), np.array(self.targets)
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(known.index(targets[i]))
        self.data, self.targets = np.squeeze(np.take(datas, mask, axis=0)), np.array(new_targets)

class CIFAR100_OSR(object):
    def __init__(self, known, dataroot='./data/cifar100', use_gpu=True, num_workers=8, batch_size=128, img_size=32):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 100))) - set(known))

        print('Selected Labels: ', known)

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False
        
        testset = CIFAR100_Filter(root=dataroot, train=False, download=True, transform=transform)
        testset.__Filter__(known=self.known)
        
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )


class SVHN_Filter(SVHN):
    """SVHN Dataset.
    """
    def __Filter__(self, known):
        targets = np.array(self.labels)
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(known.index(targets[i]))
        self.data, self.labels = self.data[mask], np.array(new_targets)

class SVHN_OSR(object):
    def __init__(self, known, dataroot='./data/svhn', use_gpu=True, num_workers=8, batch_size=128, img_size=32):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 10))) - set(known))

        print('Selected Labels: ', known)

        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False

        trainset = SVHN_Filter(root=dataroot, split='train', download=True, transform=train_transform)
        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known)
        
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        
        testset = SVHN_Filter(root=dataroot, split='test', download=True, transform=transform)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)
        
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        outset = SVHN_Filter(root=dataroot, split='test', download=True, transform=transform)
        outset.__Filter__(known=self.unknown)

        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))

class Tiny_ImageNet_Filter(ImageFolder):
    """Tiny_ImageNet Dataset.
    """
    def __Filter__(self, known):
        datas, targets = self.imgs, self.targets
        new_datas, new_targets = [], []
        for i in range(len(datas)):
            if datas[i][1] in known:
                new_item = (datas[i][0], known.index(datas[i][1]))
                new_datas.append(new_item)
                # new_targets.append(targets[i])
                new_targets.append(known.index(targets[i]))
        datas, targets = new_datas, new_targets
        self.samples, self.imgs, self.targets = datas, datas, targets

class Tiny_ImageNet_OSR(object):
    def __init__(self, known, dataroot='./data/tiny_imagenet', use_gpu=True, num_workers=8, batch_size=128, img_size=64):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 200))) - set(known))

        print('Selected Labels: ', known)
    
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False

        trainset = Tiny_ImageNet_Filter(os.path.join(dataroot, 'tiny-imagenet-200', 'train'), train_transform)
        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known)
        
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        
        testset = Tiny_ImageNet_Filter(os.path.join(dataroot, 'tiny-imagenet-200', 'val'), transform)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)
        
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        outset = Tiny_ImageNet_Filter(os.path.join(dataroot, 'tiny-imagenet-200', 'val'), transform)
        outset.__Filter__(known=self.unknown)

        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))

class ImageNetDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        jpeg_path = self.data[index]
        labels = self.labels[index]
        image = Image.open(jpeg_path).convert('RGB')
    
        if self.transform is not None:
            image = self.transform(image)
        labels = torch.as_tensor(int(labels), dtype = torch.int64)
        return image, labels
    
    def remove_negative_label(self):
        self.data = self.data[self.labels > -1]
        self.labels = self.labels[self.labels > -1]

    def remain_negative_label(self):
        self.data = self.data[self.labels == -2]
        self.labels = self.labels[self.labels == -2]

        
    def __len__(self):
        return len(self.data)
class ImageNet1K_OSR(object):
    def __init__(self, datasplit, dataroot='./data', use_gpu=True, num_workers=8, batch_size=128, img_size=224, few_shot = 0, cfg = None):
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))
        train_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize
        ])

        json_file_path = os.path.join(dataroot, 'ImageNet1K', 'protocols/imagenet_class_index.json')
        with open(json_file_path, 'r') as f:
            name_dic = json.load(f)
        clean_names = np.load(os.path.join(dataroot, 'ImageNet1K', 'protocols', 'imagenet_class_clean.npy'))
        filedir_name = {}
        t_for_clean_change = 0
        for k, v in name_dic.items():
            filedir_name[v[0]] = clean_names[t_for_clean_change]
            t_for_clean_change += 1
        csv_name_dic = {"ImageNet1k_p1":'p1', 'ImageNet1k_p2':'p2', 'ImageNet1k_p3':'p3', 'ImageNet1k_p4':'p4', 'ImageNet1k_p5':'p5', 'ImageNet1k_p6':'p6',
                        "ImageNet1k_p7":'p7', 'ImageNet1k_p8':'p8', 'ImageNet1k_p9':'p9', 'ImageNet1k_p10':'p10', 'ImageNet1k_p11':'p11', 'ImageNet1k_p12':'p12'}
        train_file_path = os.path.join(dataroot, 'ImageNet1K', 'protocols', csv_name_dic[datasplit] + '_train.csv')
        test_file_path = os.path.join(dataroot, 'ImageNet1K', 'protocols', csv_name_dic[datasplit] + '_test.csv')
        train_set = pd.read_csv(train_file_path)
        test_set = pd.read_csv(test_file_path)
        
        dir_label = train_set.groupby(train_set.iloc[:,0].str.split('/').str[1]).first().iloc[:,1].to_dict()
        key_list = [k for k, v in sorted(dir_label.items(), key=lambda item: item[1]) if v > -1]
        known = [filedir_name[fileid] for fileid in key_list]
        key_list_open_vocabulary = key_list[:math.ceil(len(key_list) // 10)]
        known_open_vocabulary = [filedir_name[fileid] for fileid in key_list_open_vocabulary]

        train_set_open_vocabulary = train_set[train_set.iloc[:,0].str.split('/').str[1].isin(key_list_open_vocabulary)]
        all_test_set_names = []
        for x in test_set.iloc[:,1]:
            if x < 0:
                all_test_set_names.append("nothing")
            else:
                all_test_set_names.append(name_dic[str(x)][0])
        all_test_set_names = pd.Series(all_test_set_names)
        test_set_open_vocabulary = test_set[all_test_set_names.isin(key_list_open_vocabulary)]
        if datasplit in ["ImageNet1k_p1", "ImageNet1k_p2", "ImageNet1k_p3"]:
            test_set_open_vocabulary = test_set[test_set.iloc[:,0].str.split('/').str[1].isin(key_list_open_vocabulary)]
        # test_set_open_vocabulary = test_set[[name_dic[x][0] for x in test_set.iloc[:,1]].isin(key_list_open_vocabulary)]
        
        image_root = dataroot + '/ImageNet1K/ILSVRC/Data/CLS-LOC/'

        train_set.iloc[:,0] = train_set.iloc[:,0].apply(lambda x: f"{image_root}{x}")
        test_set.iloc[:,0] = test_set.iloc[:,0].apply(lambda x: f"{image_root}{re.sub(r'n[0-9]*/','', x)}")
        
        train_set_open_vocabulary.iloc[:,0] = train_set_open_vocabulary.iloc[:,0].apply(lambda x: f"{image_root}{x}")
        test_set_open_vocabulary.iloc[:,0] = test_set_open_vocabulary.iloc[:,0].apply(lambda x: f"{image_root}{re.sub(r'n[0-9]*/','', x)}")
        Trainset_open_vocabulary = ImageNetDataset(np.array(train_set_open_vocabulary.iloc[:,0]), np.array(train_set_open_vocabulary.iloc[:,1]), transform=train_transform)
        Testset_open_vocabulary = ImageNetDataset(np.array(test_set_open_vocabulary.iloc[:,0]), np.array(test_set_open_vocabulary.iloc[:,1]), transform=transform)
        
        Trainset = ImageNetDataset(np.array(train_set.iloc[:,0]), np.array(train_set.iloc[:,1]), transform=train_transform)
        Testset = ImageNetDataset(np.array(test_set.iloc[:,0]), np.array(test_set.iloc[:,1]), transform=transform)
        Outset = ImageNetDataset(np.array(test_set.iloc[:,0]), np.array(test_set.iloc[:,1]), transform=transform)
        
        Trainset_open_vocabulary.remove_negative_label()
        Testset_open_vocabulary.remove_negative_label()
        Trainset.remove_negative_label()
        Testset.remove_negative_label()
        Outset.remain_negative_label()
        
        FewshotSampler_Trainset = FewshotRandomSampler(Trainset, num_samples_per_class=few_shot)
        FewshotSampler_Trainset_open_vocabulary = FewshotRandomSampler(Trainset_open_vocabulary, num_samples_per_class=few_shot)
    
        self.num_classes = len(known)
        self.known = known
        self.classes = known
        
        self.num_classes_open_vocabulary = len(known_open_vocabulary)
        self.known_open_vocabulary = known_open_vocabulary
        self.classes_open_vocabulary = known_open_vocabulary
        
        print('Selected Labels: ', known)

        pin_memory = True if use_gpu else False

        print('All Train Data:', len(Trainset))
        
        self.train_loader = torch.utils.data.DataLoader(
            Trainset, batch_size=batch_size,
            num_workers=num_workers, pin_memory=pin_memory, sampler=FewshotSampler_Trainset
        )
        
        self.train_loader_open_vocabulary = torch.utils.data.DataLoader(
            Trainset_open_vocabulary, batch_size=batch_size,
            num_workers=num_workers, pin_memory=pin_memory, sampler=FewshotSampler_Trainset_open_vocabulary
        )
        self.test_loader_open_vocabulary = torch.utils.data.DataLoader(
            Testset_open_vocabulary, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        print('All Test Data:', len(Testset))
        self.test_loader = torch.utils.data.DataLoader(
            Testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.out_loader = torch.utils.data.DataLoader(
            Outset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train: ', len(Trainset), 'Test: ', len(Testset), 'Out: ', len(Outset))
        print('All Test: ', (len(Testset) + len(Outset)))
        
        if cfg['stage'] <= 2:
            self.train_loader = self.train_loader_open_vocabulary
            self.test_loader = self.test_loader_open_vocabulary
            self.num_classes = self.num_classes_open_vocabulary
            self.known = self.known_open_vocabulary
            self.classes = self.classes_open_vocabulary
        

class ImageNet_OOD(object):
    def __init__(self, ood_dataset, dataroot='./data', use_gpu=True, num_workers=8, batch_size=128, img_size=224, shot = 0):
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))
        train_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize
        ])

        json_file_path = os.path.join(dataroot, 'ImageNet1K', 'protocols/imagenet_class_index.json')
        with open(json_file_path, 'r') as f:
            name_dic = json.load(f)
        clean_names = np.load(os.path.join(dataroot, 'ImageNet1K', 'protocols', 'imagenet_class_clean.npy'))
        filedir_name = {}
        t_for_clean_change = 0
        for k, v in name_dic.items():
            filedir_name[v[0]] = clean_names[t_for_clean_change]
            t_for_clean_change += 1   
        train_file_path = os.path.join(dataroot, 'ImageNet1K', 'protocols',  'ood_train.csv')
        test_file_path = os.path.join(dataroot, 'ImageNet1K', 'protocols',  'ood_test.csv')
        train_set = pd.read_csv(train_file_path)
        test_set = pd.read_csv(test_file_path)
        
        dir_label = train_set.groupby(train_set.iloc[:,0].str.split('/').str[1]).first().iloc[:,1].to_dict()
        key_list = [k for k, v in sorted(dir_label.items(), key=lambda item: item[1]) if v > -1]
        known = [filedir_name[fileid] for fileid in key_list]
        
        image_root = dataroot + '/ImageNet1K/ILSVRC/Data/CLS-LOC/'
        train_set.iloc[:,0] = train_set.iloc[:,0].apply(lambda x: f"{image_root}{x}")
        test_set.iloc[:,0] = test_set.iloc[:,0].apply(lambda x: f"{image_root}{re.sub(r'n[0-9]*/','', x)}")
        Trainset = ImageNetDataset(np.array(train_set.iloc[:,0]), np.array(train_set.iloc[:,1]), transform=train_transform)
        Testset = ImageNetDataset(np.array(test_set.iloc[:,0]), np.array(test_set.iloc[:,1]), transform=transform)
        FewshotSampler = FewshotRandomSampler(Trainset, num_samples_per_class=shot)
        print("ood_dataset = ", ood_dataset)
        if ood_dataset == 'iNaturalist':
            Outset = ImageFolder(root=os.path.join(dataroot, 'iNaturalist'), transform=transform)
        elif ood_dataset == 'SUN':
            Outset = ImageFolder(root=os.path.join(dataroot, 'SUN'), transform=transform)
        elif ood_dataset == 'places365': # filtered places
            Outset = ImageFolder(root= os.path.join(dataroot, 'Places'),transform=transform)  
        elif ood_dataset == 'dtd':
            Outset = ImageFolder(root=os.path.join(dataroot, 'dtd', 'images'),transform=transform)
        elif ood_dataset == 'NINCO':
            Outset = ImageFolder(root=os.path.join(dataroot, 'NINCO', 'NINCO_OOD_classes'),transform=transform)
        elif ood_dataset == 'ImageNet-O':
            Outset = ImageFolder(root=os.path.join(dataroot, 'imagenet-o'),transform=transform)
        elif ood_dataset == 'ImageNet-1K-OOD':
            Outset = ImageFolder(root=os.path.join(dataroot, 'imagenet-1k-ood'), transform=transform)
        elif ood_dataset == 'OpenImage-O':
            Outset = ImageFolder(root=os.path.join(dataroot, 'OpenImage-O'), transform=transform)
        else:
            print("Wrong ood dataset!!")
        
        self.num_classes = len(known)
        self.known = known
        pin_memory = True if use_gpu else False

        print('All Train Data:', len(Trainset))
        
        self.train_loader = torch.utils.data.DataLoader(
            Trainset, batch_size=batch_size,
            num_workers=num_workers, pin_memory=pin_memory, sampler=FewshotSampler
        )
        
        print('All Test Data:', len(Testset))
        self.test_loader = torch.utils.data.DataLoader(
            Testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.out_loader = torch.utils.data.DataLoader(
            Outset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train: ', len(Trainset), 'Test: ', len(Testset), 'Out: ', len(Outset))
        print('All Test: ', (len(Testset) + len(Outset)))

class FewshotRandomSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_samples_per_class):
        self.dataset = dataset
        self.labels = self.dataset.labels
        self.class_counts = np.bincount(self.labels)
        self.num_samples_per_class = num_samples_per_class
        self.indices = self._get_indices()
        
    def _get_indices(self):
        indices = []
        for class_label in np.unique(self.labels):
            class_indices = np.where(self.labels == class_label)[0]
            if self.num_samples_per_class <= 0:
                # print(self.class_counts[class_label])
                class_indices = np.random.choice(class_indices, size=self.class_counts[class_label], replace=False) 
            else:
                class_indices = np.random.choice(class_indices, size=self.num_samples_per_class, replace=False)
            
            indices.extend(class_indices.tolist())
        indices = np.random.permutation(indices)
        return indices
    
    def __iter__(self):
        return iter(self.indices)
    
    def __len__(self):
        return len(self.indices)


