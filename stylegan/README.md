# Style GAN

StyleGAN is a state-of-the-art generative model developed by NVIDIA that allows for the high-fidelity synthesis of realistic images. It employs a GAN framework, which consists of a generator and a discriminator, to produce images that resemble those in a given dataset. One of its key features is the manipulation of both global structure and fine details separately, providing more control over the generated images.

## Key Components

**Generator:** Comprises a series of mapping and synthesis layers. It takes a random noise vector and generates an image.

**Discriminator:** Evaluates the authenticity of generated images compared to real ones.

**Mapping Network:** Transforms the input latent vector into an intermediate latent space that controls the styles of the generated images.

**Style Mixing:** Allows for the manipulation and combination of different styles in the generation process.

### Generator Architecture

![Generator](assets/stylegan-generator.png)

The generator in StyleGAN consists of several blocks:

**Mapping Network:** Converts an input latent vector (typically a Gaussian distribution) into an intermediate latent space called "style space."

**Synthesis Network:** Utilizes a series of synthesis blocks that gradually transform the learned styles into an image.

**AdaIN (Adaptive Instance Normalization):** Applies learned style information to the intermediate feature maps in the synthesis process, allowing for style manipulation at different levels.

### Discriminator Architecture
The discriminator aims to distinguish between real and generated images. It comprises convolutional layers with increasing complexity to analyze images at multiple scales.

#### Dataset

For training the network, [Anime Face Dataset](https://www.kaggle.com/datasets/splcher/animefacedataset) for Kaggle was used.