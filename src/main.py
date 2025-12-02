import torch
import torchvision
import os

from dataset import HybridImageDataset
from model import HybridImageModel

if __name__ == "__main__":
    data_root = 'data'
    cf_file = 'cutoff_frequencies.txt'

    model = HybridImageModel()
    dataset = HybridImageDataset(data_root, cf_file)
    dataloader = torch.utils.data.DataLoader(dataset)

    data_iter = iter(dataloader)
    
    for i in range(len(dataset)):
        image_a, image_b, cutoff_frequency = next(data_iter)
        
        low_frequencies, high_frequencies, hybrid_image = model(image_a, image_b, cutoff_frequency)

        torchvision.utils.save_image(hybrid_image, 'results/%d_hybrid_image.jpg' % i)