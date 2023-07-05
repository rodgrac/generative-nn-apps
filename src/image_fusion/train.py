import sys
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

import matplotlib.pyplot as plt

sys.path.append('../')
from image_fusion.dataset import VisibleThermalDataset
from image_fusion import model
from utils import display_utils

if __name__ == '__main__':
    data_transforms = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    ds = VisibleThermalDataset('../../datasets/FLIR_ADAS_v2', transforms=data_transforms, subset_type='video') 

    train_size = int(0.8 * len(ds))
    val_size = len(ds) - train_size
    train_set, val_set = random_split(ds, [train_size, val_size])
    
    train_dl = DataLoader(train_set, batch_size=4, shuffle=True, pin_memory=True)
    val_dl = DataLoader(val_set, batch_size=4, shuffle=False, pin_memory=True)
    
    dataloaders = {'train': train_dl,
                   'val': val_dl}
    
    dataset_sizes = {'train': len(train_set),
                     'val': len(val_set)}
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Viz first batch
    batch = next(iter(dataloaders['train']))
    display_utils.visualize_batch(batch[0])  # visible
    display_utils.visualize_batch(batch[1])  # thermal
    
    plt.show()

    
    
    
    
    