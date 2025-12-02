import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kernel import create_Gaussian_kernel_2D


class HybridImageModel(nn.Module):
    def __init__(self):
        super(HybridImageModel, self).__init__()

    def get_kernel(self, cutoff_frequency: int) -> torch.Tensor:
        """
        Returns a Gaussian kernel using the specified cutoff frequency.

        Args
            cutoff_frequency: int specifying cutoff_frequency
        Returns
            kernel: Tensor of shape (c, 1, k, k) where c is # channels
        """
        
        kernel = create_Gaussian_kernel_2D(int(cutoff_frequency))
        kernel = kernel[np.newaxis, np.newaxis, :, :]
        kernel = np.tile(kernel, (self.n_channels, 1, 1, 1))
        kernel = torch.tensor(kernel, dtype=torch.float32)
        return kernel

    def low_pass(self, x: torch.Tensor, kernel: torch.Tensor):
        """
        Applies low pass filter to the input image.

        Args:
            x: Tensor of shape (b, c, m, n) where b is batch size
            kernel: low pass filter to be applied to the image
        Returns:
            filtered_image: Tensor of shape (b, c, m, n)
        """
        filtered_image = F.conv2d(input=x, weight=kernel,
                                  padding=kernel.shape[-1]//2, groups=self.n_channels)
        
        return filtered_image

    def forward(
        self, image1: torch.Tensor, image2: torch.Tensor, cutoff_frequency: torch.Tensor
    ):
        """
        Takes two images and creates a hybrid image. Returns the low frequency
        content of image1, the high frequency content of image 2, and the
        hybrid image.

        Args:
            image1: Tensor of shape (b, c, m, n)
            image2: Tensor of shape (b, c, m, n)
            cutoff_frequency: Tensor of shape (b)
        Returns:
            low_frequencies: Tensor of shape (b, c, m, n)
            high_frequencies: Tensor of shape (b, c, m, n)
            hybrid_image: Tensor of shape (b, c, m, n)
        """
        self.n_channels = image1.shape[1]
        frequency_a, frequency_b = cutoff_frequency[0]
        kernel_a = self.get_kernel(frequency_a)
        low_frequencies = self.low_pass(image1, kernel_a)

        kernel_b = self.get_kernel(frequency_b)
        high_frequencies = image2 - self.low_pass(image2, kernel_b)

        hybrid_image = torch.clamp(low_frequencies + high_frequencies, min=0, max=1)

        return low_frequencies, high_frequencies, hybrid_image
