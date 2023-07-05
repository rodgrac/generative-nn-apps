import sys
import torch
import time
from datetime import datetime
from tqdm import tqdm

from torch.utils.data import DataLoader, random_split
from torchvision import transforms

sys.path.append('../')
from image_fusion.dataset import VisibleThermalDataset
from image_fusion.model import BimodalAutoEncoder


torch.manual_seed(42)


def run_train(model, dataloaders, params, device, epochs):
    since = time.time()
    
    for epoch in range(epochs):
        print(f'Epoch {epoch}/{epochs-1}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0
            
            for input1, input2 in tqdm(dataloaders[phase]):
                input1 = input1.to(device)
                input2 = input2.to(device)
                
                params['opt'].zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    out1, out2 = model(input1, input2)
                    loss = params['loss'](out1, input1) + params['loss'](out2, input2)
                    
                    if phase == 'train':
                        loss.backward()
                        params['opt'].step()
                
                running_loss += loss.item() * input1.size(0)
                
            if phase == 'train':
                params['sched'].step()
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print(f'{phase} Loss: {epoch_loss:.4f}')
            
    time_elapsed = time.time() - since
    
    print(f'Training done in {(time_elapsed//60):.0f}m {(time_elapsed%60):.0f}s')
    
    return model  


if __name__ == '__main__':
    
    data_transforms = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    batch_size = 64
    n_epochs = 10
    lr = 0.001

    ds = VisibleThermalDataset('../../datasets/FLIR_ADAS_v2', transforms=data_transforms, subset_type='video') 

    train_size = int(0.8 * len(ds))
    val_size = len(ds) - train_size
    train_set, val_set = random_split(ds, [train_size, val_size])
    
    print(f'Train samples: {len(train_set)} Val samples: {len(val_set)}')
    
    train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_dl = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    dataloaders = {'train': train_dl,
                   'val': val_dl}
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')
    
    model = BimodalAutoEncoder().to(device)

    params = dict()
    params['loss'] = torch.nn.MSELoss()
    params['opt'] = torch.optim.Adam(model.parameters(), lr=lr)
    params['sched'] = torch.optim.lr_scheduler.StepLR(params['opt'], step_size=5, gamma=0.01)
            
    # Viz first batch
    # batch = next(iter(dataloaders['train']))
    # display_utils.visualize_batch(batch[0])  # visible
    # display_utils.visualize_batch(batch[1])  # thermal
    # plt.show()
    # exit()
    
    model = run_train(model, dataloaders, params, device, n_epochs)
    
    now = datetime.now()
    torch.save(model, f'../../out/model_image_fusion_{now.strftime("%Y-%m-%d-%H-%M-%S")}.pt')
    
                  
    
    
    
    
    