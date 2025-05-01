
import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def get_augemented_dataloader(data_path="data/food-101", img_size=64, batch_size=32):

    os.chdir("../..")
    test_data_path = os.path.join(data_path, "test")
    train_data_path = os.path.join(data_path, "train")

    train_transforms_trivial_augment = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor()
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor()
    ])
    
    train_data_augmented = ImageFolder(train_data_path, transform=train_transforms_trivial_augment)
    test_data_augmented = ImageFolder(test_data_path, transform=test_transforms)

    train_dataloader_aug = DataLoader(dataset=train_data_augmented, batch_size=batch_size, shuffle=True)
    test_dataloader_aug = DataLoader(dataset=test_data_augmented, batch_size=batch_size, shuffle=True)

    return train_dataloader_aug, test_dataloader_aug
