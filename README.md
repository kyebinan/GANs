# GANs
Different Generative Adversarial Networks (GANs) detailed Architecture Understanding

# Introduction
Welcome to a comprehensive exploration of various Generative Adversarial Networks (GANs) architectures. This repository delves into the intricacies of different GAN models, shedding light on their unique architectures and applications in the realm of generative deep learning.

## 1-  Generative Adversarial Networks (GAN)

## 2- Deep Convolutional Generative Adversarial Networks (DCGAN)
Inspired by the groundbreaking paper [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://github.com/kyebinan/GANs/blob/main/papers/1511.06434.pdf)  by Alec Radford, Luke Metz, and Soumith Chintala, this project aims to provide a clear and educational implementation of a DCGAN for generating realistic images. DCGANs have demonstrated remarkable success in generating high-quality, diverse images, making them a cornerstone in the field of generative deep learning. In this repository, you'll find the Python code for training a DCGAN model, along with explanations, comments, and potential extensions for further exploration. As an exciting twist, I'll be using this DCGAN to generate images of Pokémon, adding a playful and imaginative element to the project. I'll let you look at the project's [Notebook](https://github.com/kyebinan/GANs/blob/main/DCGAN/nb_dcgan.ipynb).
Whether you're a beginner looking to understand the fundamentals of GANs or an enthusiast eager to experiment with your own image generation, this implementation is designed to be both informative and accessible. Join me on this journey into the world of generative adversarial networks, and let's explore the fascinating realm of artificial creativity together. Feel free to contribute, ask questions, or use this code as a starting point for your own projects. Happy coding!

<p align="center">
  <img src="https://github.com/kyebinan/GANs/assets/155234248/536085cb-2d3b-416e-a76b-22b05d280a2b" width="400" height="400" alt="MNIST Fake"/>
  <img src="https://github.com/kyebinan/GANs/assets/155234248/0425c44c-e2ec-453e-a00f-a4b1672c1447" width="400" height="400" alt="MNIST Real"/>
</p>
<p align="center">
  <img src="https://github.com/kyebinan/GANs/assets/155234248/dcfbf215-e76f-4a5e-bda2-94097ac9c98f" width="400" height="400" alt="Pokemon Fake"/>
  <img src="https://github.com/kyebinan/GANs/assets/155234248/75b651b3-5390-4a2c-87d5-3d15c9e66845" width="400" height="400" alt="Pokemon Real"/>
</p>

## 3- Wasserstein Generative Adversarial Networks (Wasserstein GANs)
Wasserstein Generative Adversarial Networks (Wasserstein GANs) stand out for their unique discriminator training strategy, which involves minimizing the Wasserstein distance between the generated and real data distributions. Proposed by Martin Arjovsky, Soumith Chintala, and Léon Bottou in the paper "Wasserstein GAN," this approach addresses some of the training instability issues associated with traditional GANs. Unlike traditional GANs, Wasserstein GANs provide a more meaningful and stable training signal by avoiding issues like mode collapse. The section dedicated to Wasserstein GANs in this repository explores their underlying architecture, training principles, and applications. Whether you're seeking a more stable alternative to traditional GANs or aiming to understand the Wasserstein distance concept, Wasserstein GANs offer a valuable perspective in the landscape of generative adversarial networks. Dive into the code, explore Wasserstein GANs, and discover their potential in generating high-quality, diverse samples.

### a- WGAN
What it is: The Wasserstein GAN (WGAN) introduces a novel way to measure the distance between the generator's distribution and the real data distribution using the Wasserstein distance, which helps in improving the stability of the training process and addresses issues like mode collapse seen in traditional GANs.

Pros:
Improved Stability: WGANs offer more stable training compared to traditional GANs, making it easier to train models without worrying about issues like vanishing gradients.
Better Convergence Metrics: The Wasserstein distance provides a more meaningful measurement of the difference between the generated and real data distributions, allowing for more interpretable training progress.

Cons:
Computational Cost: The computation of the Wasserstein distance can be more demanding, leading to longer training times.
Hyperparameter Sensitivity: WGANs can be sensitive to the choice of hyperparameters, such as the clipping parameter used to enforce the Lipschitz constraint, requiring careful tuning.

<p align="center">
  <img src="https://github.com/kyebinan/GANs/assets/155234248/403a0980-85fa-4eca-8f49-28ba66ccf627" width="400" height="400" alt="MNIST Fake"/>
  <img src="https://github.com/kyebinan/GANs/assets/155234248/b1b4f4fb-da39-4cc7-92cf-ab920e328673" width="400" height="400" alt="MNIST Real"/>
</p>

### b- WGAN with Gradient Penalty  (WGAN-GP)
What it is: WGAN-GP builds on the WGAN framework by introducing a gradient penalty term in the loss function to enforce the Lipschitz constraint more effectively, which further stabilizes training and improves the quality of generated samples.

Pros:
Enhanced Stability and Sample Quality: The gradient penalty helps in further stabilizing the training of WGANs and often results in higher quality samples by ensuring a smoother gradient flow.
Eliminates Weight Clipping: By using a gradient penalty instead of weight clipping to enforce the Lipschitz constraint, WGAN-GP avoids potential issues like capacity underuse and exploding/vanishing gradients.

Cons:
Increased Computational Complexity: The introduction of the gradient penalty term increases the computational cost since it requires additional gradient computations during training.
Tuning Required: While WGAN-GP reduces some of the hyperparameter sensitivity seen in WGAN, it introduces its own hyperparameters (like the penalty coefficient) that require careful tuning to achieve optimal performance.

https://github.com/kyebinan/GANs/assets/155234248/293d2145-1904-4b07-bf2b-6ae5e050a4c6

## 4- Conditional Generative Adversarial Networks (Conditional GAN)
Conditional Generative Adversarial Networks, or Conditional GANs, represent a powerful variant of GANs designed to introduce control and directionality into the generative process. Unlike traditional GANs that generate data without specific constraints, Conditional GANs incorporate additional information, often in the form of labels or auxiliary data, to guide the generation process. This conditional input allows users to influence the characteristics of the generated samples. In the original Conditional GAN paper, titled "Conditional Generative Adversarial Nets" by Mehdi Mirza and Simon Osindero, the authors propose a novel architecture where both the generator and discriminator receive additional conditioning information. This conditioning can be used for tasks such as image-to-image translation, style transfer, and generating samples from specific classes. This section of the repository provides a detailed exploration of Conditional GANs, including their architecture, training process, and practical applications. Whether you're interested in adding specific attributes to generated images or exploring the intersection of GANs and conditional modeling, Conditional GANs offer an exciting avenue for experimentation and creative exploration. Dive into the code, understand the conditional input concept, and unlock the potential of conditional generative modeling.



## 5- Bicycle Generative Adversarial Network (BiCycleGAN)
BiCycleGAN, short for Bicycle Generative Adversarial Network, is an intriguing extension of the traditional GAN architecture, aiming to enhance the generator's capability to generate diverse outputs while maintaining a certain level of control. Introduced in the paper "Toward Multimodal Image-to-Image Translation" by Jun-Yan Zhu et al., BiCycleGAN introduces a bidirectional consistency loss, where the generator not only generates images from input data but also reconstructs the original input from the generated output. This bidirectional mapping helps enforce a more structured and controllable image-to-image translation process. The generator is trained to not only generate diverse outputs but also ensure that these outputs can be accurately reconstructed back to the original input. This added constraint contributes to improved stability and controllability in the generated results. The repository associated with this section provides an implementation of BiCycleGAN, along with insights into its architecture, training process, and potential applications. Whether you're interested in multimodal image translation or wish to explore advanced GAN architectures, BiCycleGAN offers a fascinating avenue for experimentation and learning. Dive into the code, explore the bidirectional consistency concept, and discover the unique capabilities of BiCycleGAN.


## 6- Cycle-Consistent Generative Adversarial Network (CycleGAN)
CycleGAN, short for Cycle-Consistent Generative Adversarial Network, introduces a novel approach to unsupervised image-to-image translation without the need for paired training data. Proposed by Jun-Yan Zhu et al. in the paper "Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks," CycleGAN leverages the idea of cycle consistency, where the translation from domain A to domain B and back from domain B to domain A should ideally return the original input. This unique constraint enforces a cycle-consistent mapping, enhancing the model's ability to translate images between domains while preserving essential characteristics. The repository section dedicated to CycleGAN provides an in-depth exploration of its architecture, training dynamics, and practical applications. Whether you're interested in style transfer, domain adaptation, or exploring the potential of unsupervised image translation, CycleGAN offers a versatile and powerful solution. Delve into the code, understand the concept of cycle consistency, and embark on a journey of image translation without paired data.


