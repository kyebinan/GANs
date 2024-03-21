"""
Discriminator and Generator implementation from DCGAN paper
"""

import torch 
import torch.nn as nn


class Discriminator(nn.Module):
    """
    A Discriminator class for a Generative Adversarial Network (GAN), implemented as a PyTorch module.

    This class defines a discriminator network that classifies images as real or fake. It is structured as a 
    convolutional neural network (CNN) with LeakyReLU activations and batch normalization. The architecture is 
    designed to process images of size 64x64 pixels with a specified number of channels.

    Parameters:
    - channels_img (int): The number of channels in the input images. For example, 3 for RGB images.
    - features_d (int): The base number of features in the discriminator network. This number of features 
      is scaled up in deeper layers of the network.

    The discriminator network consists of multiple convolutional layers, where each layer (except for the first 
    and last) is preceded by a block that includes convolution, batch normalization, and LeakyReLU activation. 
    The final layer uses a sigmoid activation function to output a probability indicating whether the input image 
    is real or fake.

    The `_block` method defines a repeated structure used in the network, consisting of a convolutional layer 
    followed by batch normalization and LeakyReLU activation.

    The `forward` method defines the forward pass of the network, taking an input tensor `x` and passing it through 
    the discriminator network to produce the output probability.

    Example usage:
    ```
    # Initialize discriminator
    discriminator = Discriminator(channels_img=3, features_d=64)

    # Forward pass with an input image tensor of size (N, 3, 64, 64)
    output = discriminator(image_tensor)
    ```

    Note: This class is intended to be used within the context of training GANs, where it serves as the discriminator 
    component. It is typically trained alongside a generator network in an adversarial setup.
    """
    def __init__(self, channels_img, features_d):
        super().__init__()
        self.disc = nn.Sequential(
            # Input : N x channels_img x 64 x 64
            nn.Conv2d(
                channels_img, features_d, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d*2, 4, 2, 1),   # 16x16
            self._block(features_d*2, features_d*4, 4, 2, 1), # 8x8
            self._block(features_d*4, features_d*8, 4, 2, 1), # 4x4  
            nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=0), #1x1
        )


    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            #nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, x):
        return self.disc(x)
    


class Generator(nn.Module):
    """
    A Generator class for a Generative Adversarial Network (GAN), implemented as a PyTorch module.

    This class defines a generator network that generates images from noise vectors. It utilizes a series 
    of transposed convolutional layers (also known as deconvolutional layers) with ReLU activations and batch 
    normalization to produce images of a target size and channel depth. The architecture is specifically 
    designed to upscale a latent noise vector into a 64x64 pixel image with a specified number of output 
    channels (e.g., 3 for RGB images).

    Parameters:
    - z_dim (int): The dimensionality of the input latent noise vector.
    - channels_img (int): The number of channels in the output images. For example, 3 for RGB images.
    - features_g (int): The base number of features in the generator network. This number of features 
      is scaled down in deeper layers of the network to match the desired output size and channel depth.

    The generator network starts with a dense layer that reshapes the latent vector into a small feature map 
    (e.g., 4x4 pixels) and progressively upscales it to the target image size through a series of transposed 
    convolutional blocks. Each block, except for the last, consists of a transposed convolutional layer 
    followed by batch normalization and ReLU activation. The final layer uses a tanh activation function 
    to normalize the output image pixel values to the range [-1, 1].

    The `_block` method defines a repeated structure used in the network, consisting of a transposed 
    convolutional layer followed by batch normalization and ReLU activation.

    The `forward` method defines the forward pass of the network, taking an input latent vector `x` and 
    passing it through the generator network to produce an output image.

    Example usage:
    ```
    # Initialize generator
    generator = Generator(z_dim=100, channels_img=3, features_g=64)

    # Generate an image from a random noise vector of size (N, z_dim, 1, 1)
    noise = torch.randn((N, z_dim, 1, 1))
    fake_image = generator(noise)
    ```

    Note: This class is intended to be used within the context of training GANs, where it serves as the 
    generator component. It is typically trained in an adversarial setup against a discriminator network.
    """
    def __init__(self, z_dim, channels_img, features_g):
        super().__init__()
        self.gen = nn.Sequential(
            self._block(z_dim, features_g*16, 4, 1, 0), # N x f_g*16 x 4 x 4
            self._block(features_g*16, features_g*8, 4, 2, 1), # 8x8 
            self._block(features_g*8, features_g*4, 4, 2, 1), # 16x16 
            self._block(features_g*4, features_g*2, 4, 2, 1), # 32x32
            nn.ConvTranspose2d(
                features_g*2, channels_img, kernel_size=4, stride=2, padding=1,
            ),
            nn.Tanh(), # [-1, 1]
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.gen(x)
    
def initialize_weights(model):
    """
    Initializes the weights of convolutional and batch normalization layers in a PyTorch model 
    using a normal distribution with mean 0.0 and standard deviation 0.02. This initialization 
    is commonly used in Generative Adversarial Networks (GANs) and other deep learning models 
    to help stabilize training by setting the weights to small random values close to zero (from the paper).

    Parameters:
    - model (torch.nn.Module): The PyTorch model whose weights need to be initialized.

    This function iterates over all modules in the provided model. If a module is an instance 
    of either nn.Conv2d, nn.ConvTranspose2d, or nn.BatchNorm2d, its weights are initialized 
    with a normal distribution using the specified mean and standard deviation. Other types 
    of layers/modules are not affected by this function.

    Example usage:
    ```
    # Initialize a GAN generator model
    generator = Generator(z_dim=100, channels_img=3, features_g=64)
    
    # Initialize its weights
    initialize_weights(generator)
    ```

    Note: This function is intended to be used once at the beginning of the training process, 
    before training starts. It can be applied to both generator and discriminator models in 
    the context of GANs or to any PyTorch model that benefits from this type of weight initialization.
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def test():
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    initialize_weights(disc)
    assert disc(x).shape == (N, 1, 1, 1)

    gen = Generator(z_dim, in_channels, 8)
    initialize_weights(gen)
    z = torch.randn((N, z_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W)
    print("Success")



if __name__ == '__main__':
    test()






