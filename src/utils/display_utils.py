import numpy as np
import torchvision

import matplotlib.pyplot as plt

from utils import constants


def visualize_batch(batch_data, norm='imgnet'):
    data_grid = torchvision.utils.make_grid(batch_data)
    data_np = data_grid.numpy().transpose((1, 2, 0))
    if norm == 'imgnet':
        mean = np.array(constants.imgnet_mean)
        std = np.array(constants.imgnet_std)
    else:
        mean, std = 0, 1
        
    data_np = std * data_np + mean
    data_np = np.clip(data_np, 0, 1)
    
    plt.figure()
    plt.imshow(data_np)