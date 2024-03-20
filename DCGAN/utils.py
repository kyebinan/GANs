from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import os

class PokemonDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        Custom dataset for loading Pokémon images.
        
        Parameters:
        - image_dir (str): Directory path to the images.
        - transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")  # Convert to RGB to ensure 3 channels
        
        if self.transform:
            image = self.transform(image)
        
        return image