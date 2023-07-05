from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

from utils import file_utils


class VisibleThermalDataset(Dataset):
    def __init__(self, root_path, transforms, subset_type, split='') -> None:
        super().__init__()
        self.root_path = Path(root_path)
        self.transforms = transforms
        
        self.subsets = ['rgb', 'thermal']
        self.split = split
        
        if subset_type == 'video':
            self.subset_video_map = file_utils.load_json(self.root_path / 'rgb_to_thermal_vid_map.json')
            subset_rgb_imgs_all = (self.root_path / f'video_rgb_test/data').glob('*.jpg')
            self.subset_rgb_imgs = []
            self.subset_thermal_imgs = []
            for rgb_img in subset_rgb_imgs_all:
                if str(rgb_img.name) in self.subset_video_map:
                    self.subset_rgb_imgs.append(rgb_img)
                    self.subset_thermal_imgs.append(self.root_path / f'video_thermal_test/data/{self.subset_video_map[rgb_img.name]}')
            
            
    def __getitem__(self, index):
        rgb_img_path = self.subset_rgb_imgs[index]
        thermal_img_path = self.subset_thermal_imgs[index]
        
        rgb_img = Image.open(rgb_img_path).convert("RGB")
        thermal_img = Image.open(thermal_img_path).convert("RGB")
        
        if self.transforms is not None:
            rgb_img, thermal_img = self.transforms(rgb_img), self.transforms(thermal_img)
        
        return rgb_img, thermal_img
    
    
    def __len__(self):
        return len(self.subset_rgb_imgs)