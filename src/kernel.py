import numpy as np

def create_Gaussian_kernel_1D(ksize: int, sigma: int) -> np.ndarray:
    """
    Create a 1D Gaussian kernel using the specified filter size and standard deviation.
    
    Args:
        ksize: length of kernel
        sigma: standard deviation of Gaussian distribution

    Returns:
        kernel: 1d column vector of shape (k,1)
    """
    x = np.arange(ksize)
    kernel = 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(x-int(np.floor(ksize/2)))**2/(2*sigma**2))
    kernel /= kernel.sum()
    kernel = np.reshape(kernel, (ksize, 1))
    return kernel


def create_Gaussian_kernel_2D(cutoff_frequency: int) -> np.ndarray:
    """
    Create a 2D Gaussian kernel using the specified filter size, standard
    deviation and cutoff frequency.

    Args:
        cutoff_frequency: an int controlling how much low frequency to leave in
        the image.
    Returns:
        kernel: numpy nd-array of shape (k, k)
    """
    kernel = np.outer(create_Gaussian_kernel_1D(cutoff_frequency*4 + 1, cutoff_frequency),
                      create_Gaussian_kernel_1D(cutoff_frequency*4 + 1, cutoff_frequency))
    return kernel

