import torchvision
from torch import nn
from torchvision.transforms import transforms
import torch
import torch.utils.data as torch_data

from models.resnet import resnet18
from tasks.task import Task


class Cifar10Task(Task):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))

    def load_data(self):
        self.load_cifar_data()
        
    def load_cifar_data(self):
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            self.normalize
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            self.normalize
        ])

        original_train_dataset = torchvision.datasets.CIFAR10(
            root=self.params.data_path,
            train=True,
            download=True,
            transform=transform_train)
        
        clean_size = round(self.params.clean_ratio * len(original_train_dataset))
        train_size = len(original_train_dataset) - clean_size
        self.train_dataset, self.clean_dataset = torch_data.random_split(original_train_dataset, 
                                                                         lengths=[train_size, clean_size], 
                                                                         generator=torch.Generator().manual_seed(self.params.random_seed))
        self.clean_loader = torch_data.DataLoader(self.train_dataset,
                                                  batch_size=self.params.batch_size,
                                                  shuffle=True,
                                                  num_workers=0)
        self.clean_loader = torch_data.DataLoader(self.clean_dataset,batch_size=self.params.batch_size,
            shuffle=False, num_workers=0)
        if self.params.clean_set_dataset == 'CIFAR100':
            self.clean_dataset= torchvision.datasets.CIFAR100(root=self.params.data_path, 
                                                              train=True, 
                                                              download=True, 
                                                              transform=transform_train)
            tot = len(self.clean_dataset)
            self.clean_dataset, _ = torch_data.random_split(self.clean_dataset,
                                                            lengths=[clean_size, tot-clean_size], 
                                                            generator=torch.Generator().manual_seed(self.params.random_seed))
            self.clean_loader = torch_data.DataLoader(self.clean_dataset,
                                                      batch_size=self.params.batch_size,
                                                      shuffle=True,
                                                      num_workers=0)
    
        elif self.params.clean_set_dataset == 'GTSRB':
            transform_train= transforms.Compose([transform_train, transforms.Resize((32,32))])
            self.clean_dataset= torchvision.datasets.GTSRB(root=self.params.data_path,
                                                           split='train',
                                                           download=True, 
                                                           transform=transform_train)
            tot = len(self.clean_dataset)
            self.clean_dataset, _ = torch_data.random_split(self.clean_dataset,
                                                          lengths=[clean_size, tot-clean_size], 
                                                          generator=torch.Generator().manual_seed(self.params.random_seed))
            self.clean_loader = torch_data.DataLoader(self.clean_dataset,
                                                      batch_size=self.params.batch_size,
                                                      shuffle=True,
                                                      num_workers=0)

        elif self.params.clean_set_dataset is None and self.params.clean_classes is not None:
            clean_indices = []
            for cls in self.params.clean_classes:
                for i in range(clean_size):
                    if self.clean_dataset[i][1] == cls:
                        clean_indices.append(i)
            sampler = torch_data.SubsetRandomSampler(clean_indices, generator=torch.Generator().manual_seed(self.params.random_seed))
            self.clean_loader = torch_data.DataLoader(self.clean_dataset,
                                                      batch_size=self.params.batch_size, 
                                                      sampler=sampler, 
                                                      num_workers=0)
        else:
            self.clean_loader = torch_data.DataLoader(self.clean_dataset,
                                                      batch_size=self.params.batch_size,
                                                      shuffle=True,
                                                      num_workers=0)

        self.train_loader = torch_data.DataLoader(self.train_dataset,
                                                  batch_size=self.params.batch_size,
                                                  shuffle=True,
                                                  num_workers=0)

        
            
        self.test_dataset = torchvision.datasets.CIFAR10(
            root=self.params.data_path,
            train=False,
            download=True,
            transform=transform_test)
        self.test_loader = torch_data.DataLoader(self.test_dataset,
                                                 batch_size=self.params.test_batch_size,
                                                 shuffle=False,
                                                 num_workers=0)
        self.classes = ('plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        return True
    
    def build_model(self) -> nn.Module:
        if self.params.pretrained:
            model = resnet18(pretrained=True, norm_layer=nn.Identity)

            # model is pretrained on ImageNet changing classes to CIFAR
            model.fc = nn.Linear(512, len(self.classes))
        else:
            model = resnet18(pretrained=False,
                             num_classes=len(self.classes))
        return model

