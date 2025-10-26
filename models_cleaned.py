

#install packages

import os
import random
import time
from datetime import datetime
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torchvision.transforms as transforms

import wandb
print("Torch:", torch.__version__, "Torchvision:", torchvision.__version__, "WandB:", wandb.__version__)


# In[ ]:


# ---------------------
# Cell 2: Mount Google Drive and create save dir
# ---------------------

#drive.mount('/content/drive', force_remount=True)

#DRIVE_BASE = '/content/drive/MyDrive/vgg6_cifar10'
#os.makedirs(DRIVE_BASE, exist_ok=True)
#print("Drive base:", DRIVE_BASE)


# In[ ]:


# ---------------------
# Cell 3: Config dictionary (edit hyperparams here)
# ---------------------
config = {

    "activation": "gelu",           # relu, tanh, sigmoid, silu (swish), gelu
    "optimizer": "sdg",             # sgd, nesterov, adam, adagrad, rmsprop, nadam
    "batch_size": 128,
    "epochs": 70,
    "lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "seed": 1000,
    "num_workers": 2,
    # wandb settings
    "use_wandb": True,
    "wandb_project": "vgg6-cifar10",
    "wandb_entity": None,
    # saving
    #"save_dir": DRIVE_BASE,
    # misc
    "log_interval": 100
}

# Example: change easily
# config['activation'] = 'gelu'
# config['optimizer'] = 'adam'


# In[ ]:


# ---------------------
# Cell 4: Utility functions, model, dataloaders, optimizer mapping
# ---------------------
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

class VGG6(nn.Module):
    def __init__(self, activation: str = 'relu'):
        super(VGG6, self).__init__()
        act = self._get_activation(activation)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), act,
            nn.Conv2d(64, 128, kernel_size=3, padding=1), act,
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), act,
            nn.Conv2d(256, 256, kernel_size=3, padding=1), act,
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512), act,
            nn.Linear(512, 10)
        )

    def _get_activation(self, name: str):
        name = name.lower()
        if name == 'relu': return nn.ReLU()
        if name == 'sigmoid': return nn.Sigmoid()
        if name == 'tanh': return nn.Tanh()
        if name in ('silu', 'swish'): return nn.SiLU()
        if name == 'gelu': return nn.GELU()
        return nn.ReLU()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_dataloaders(batch_size: int, num_workers: int = 2):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),   
        transforms.RandomHorizontalFlip(),      
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    num_train = len(trainset)
    indices = list(range(num_train))
    split = int(0.9 * num_train)  # 90% train, 10% val
    train_idx, val_idx = indices[:split], indices[split:]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    val_loader = DataLoader(trainset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

def get_optimizer(name: str, params, lr: float, momentum: float = 0.9, weight_decay: float = 5e-4):
    name = name.lower()
    if name == 'sgd': return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name in ('nesterov', 'nesterov-sgd'): return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    if name == 'adam': return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == 'adagrad': return optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
    if name == 'rmsprop': return optim.RMSprop(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == 'nadam':
        # PyTorch 2.x has optim.NAdam, otherwise fallback to AdamW
        try:
            return optim.NAdam(params, lr=lr, weight_decay=weight_decay)
        except Exception:
            return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)


# In[ ]:


# ---------------------
# Cell 5: Training & evaluation functions
# ---------------------
def train_epoch(model, device, loader, criterion, optimizer, epoch, use_wandb=False, log_interval=100):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if use_wandb:
            global_step = epoch * len(loader) + batch_idx
            wandb.log({
                'train/batch_loss': loss.item(),
                'train/batch_acc': 100. * predicted.eq(targets).sum().item() / targets.size(0),
                'global_step': global_step
            })

        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(inputs)}/{len(loader.sampler)}]  Loss: {loss.item():.4f}")

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

@torch.no_grad()
def evaluate(model, device, loader, criterion):
    model.eval()
    loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss += criterion(outputs, targets).item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    loss = loss / total
    acc = 100. * correct / total
    return loss, acc


# In[ ]:





# In[ ]:


# ---------------------
# Cell 6: Main train(config) function (call this to train)
# ---------------------
def train(config: Dict):
    set_seed(config.get('seed', 42))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)

    use_wandb = config.get('use_wandb', True)
    if use_wandb:
        # init wandb
        wandb.init(project=config.get('wandb_project', 'vgg6-cifar10'),
                   entity=config.get('wandb_entity', None),
                   config=config, reinit=True)
        wb_config = wandb.config
    else:
        wb_config = config

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=config['batch_size'], num_workers=config.get('num_workers', 2))

    model = VGG6(activation=config.get('activation', 'relu')).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(config.get('optimizer', 'sgd'), model.parameters(), lr=config.get('lr', 0.1),
                              momentum=config.get('momentum', 0.9), weight_decay=config.get('weight_decay', 5e-4))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    best_val_acc = 0.0
    best_state = None
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}

    start_time = time.time()
    for epoch in range(config['epochs']):
        train_loss, train_acc = train_epoch(model, device, train_loader, criterion, optimizer, epoch,
                                            use_wandb=use_wandb, log_interval=config.get('log_interval', 100))
        val_loss, val_acc = evaluate(model, device, val_loader, criterion)

        history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss); history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        print(f"Epoch {epoch}: Train Loss {train_loss:.4f} Acc {train_acc:.2f}% | Val Loss {val_loss:.4f} Acc {val_acc:.2f}% | LR {optimizer.param_groups[0]['lr']:.5f}")

        if use_wandb:
            wandb.log({
                'train/loss': train_loss, 'train/acc': train_acc,
                'val/loss': val_loss, 'val/acc': val_acc,
                'lr': optimizer.param_groups[0]['lr'], 'epoch': epoch
            })

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'config': config
            }
            os.makedirs(config['save_dir'], exist_ok=True)
            save_path = os.path.join(config['save_dir'], 'best_model.pth')
            torch.save(best_state, save_path)
            print(f"Saved best model to {save_path} (val acc: {best_val_acc:.2f}%)")
            if use_wandb:
                wandb.save(save_path)

        scheduler.step()

    elapsed = time.time() - start_time
    print(f"Training finished in {elapsed/60:.2f} minutes. Best val acc: {best_val_acc:.2f}%")

    # final test evaluation using the best saved model
    if best_state is not None:
        model.load_state_dict(best_state['model_state'])
        test_loss, test_acc = evaluate(model, device, test_loader, criterion)
        print(f"Test Loss: {test_loss:.4f} Test Acc: {test_acc:.2f}%")
        if use_wandb:
            wandb.log({'test/loss': test_loss, 'test/acc': test_acc})

    # create a small README for reproducibility (saved to Drive)
    readme_path = os.path.join(config['save_dir'], 'README_RUN.txt')
    with open(readme_path, 'w') as f:
        f.write('VGG6 CIFAR-10 training (PyTorch, Colab)\\n')
        f.write('Transforms: RandomCrop(32, padding=4), RandomHorizontalFlip, Normalize(mean,std)\\n')
        f.write('Example config:\\n' + str(config) + '\\n')
        f.write('Best model path: ' + os.path.join(config['save_dir'], 'best_model.pth') + '\\n')
    print("Wrote README to", readme_path)

    if use_wandb:
        wandb.finish()

    return history, best_state


# In[ ]:


# ---------------------

# ---------------------
# To actually run a sweep:
# Ensure you're logged in to wandb in Colab: wandb.login()


sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val/acc', 'goal': 'maximize'},
    'parameters': {
        'activation': {'values': ['relu', 'tanh', 'silu', 'gelu']},
        'optimizer': {'values': ['sgd', 'nesterov', 'adam', 'rmsprop']},
        'batch_size': {'values': [64, 128]},
        'lr': {'values': [0.1, 0.01, 0.001]},
        'epochs': {'value': 30}
    }
}


# In[ ]:


history, best_state=train(config)

