# FedGame
Official implementation for our paper "FedGame: A Game-Theoretic Defense against Backdoor Attacks in Federated Learning" (NeurIPS 2023).

## Requirements
```bash
$ pip install -r requirements.txt
$ mkdir runs
$ mkdir saved_models
```
Download Tiny ImageNet:
```bash
$ wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
```
Preprocessing Tiny ImageNet:
```python
import os

DATA_DIR = '.data/tiny-imagenet-200'  # Original images come in shapes of [3,64,64]

# Define training and validation data paths
TRAIN_DIR = os.path.join(DATA_DIR, 'train') 
VALID_DIR = os.path.join(DATA_DIR, 'val')

val_img_dir = os.path.join(VALID_DIR, 'images')

# Open and read val annotations text file
fp = open(os.path.join(VALID_DIR, 'val_annotations.txt'), 'r')
data = fp.readlines()

# Create dictionary to store img filename (word 0) and corresponding
# label (word 1) for every line in the txt file (as key value pair)
val_img_dict = {}
for line in data:
    words = line.split('\t')
    val_img_dict[words[0]] = words[1]
fp.close()

# Display first 10 entries of resulting val_img_dict dictionary
{k: val_img_dict[k] for k in list(val_img_dict)[:10]}

for img, folder in val_img_dict.items():
    newpath = (os.path.join(val_img_dir, folder))
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    if os.path.exists(os.path.join(val_img_dir, img)):
        os.rename(os.path.join(val_img_dir, img), os.path.join(newpath, img))
```

## Usage
FedAvg:
```bash
$ python training.py --name fedavg --params configs/mnist_fedavg.yaml
```
FLTrust:
```
$ python training.py --name fltrust --params configs/mnist_fltrust.yaml
```
Ours:
```bash
$ python training.py --name ours --params configs/mnist_ours.yaml
```


## Adaptation
Follow the following steps to adopt existing code to a new dataset.

### Set Parameters
**Create `configs/<your task>.yaml`.**

All paramters and their default values (if any) are listed in `utils/parameters.py`. Parameters customized for specific tasks are set in `configs/\<your task>.yaml`. If you have a new attack setting (e.g. with new datasets/algorithms), create one file with your parameters under the `configs` folder.

### Add New Dataset
**Create `tasks/<your dataset>.py` and `tasks/<your dataset for fl>.py`.**

Datasets are defined in each `Task` class. If you want to adopt our defense to `CIFAR`, first create a `cifar10_task.py` that defines how the dataset should be loaded. (This part is already done.) Then, create a file under `mnist_ours_task` that defines how to load data under FL (and of course other things). **Remember to change number of channels in the trigger defination of `reverse_engineer_per_class` if necessary.** It should be 1 for MNIST and 3 for CIFAR10.
