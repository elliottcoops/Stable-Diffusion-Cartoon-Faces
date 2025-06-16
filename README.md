# Stable-Diffusion-Cartoon-Faces

## About

This repository contains a Stable Diffusion model to generate synthetic images of cartoon faces from a given prompt, trained using Skiittoo/cartoon-faces from Hugging Face.

![VAE reconstruction](images/generated_examples.png)

## What is stable diffusion?

Stable Diffusion works by adding noise to a compressed image and then progressively denoising it with a UNet, guided by CLIP to align with the text prompt. Working directly in pixel space is computationally intensive (a 256x256x3 image contains over 196,000 pixels), so compression into a tighter latent space is preferred.

More abstractly, the model has two main components:

1. **VAE (Encoder/Decoder)** - Compresses and decompresses images.

2. **UNet (Bottleneck)** - Denoises and generates details to produce the final image.

![image](images/stable_diffusion_diagram.jpg)

## Training

Both the VAE and UNet were trained from scratch and their architectures can be found in `{unet/vae}_architecture.py`. Training was carried out on Kaggle using an NVIDIA TESLA P100.

### VAE training

The VAE was trained to encode 64x64x3 images into a spatial latent representation of size 4x4 with 4 channels. The general goal of the model during training was to minimise the:

1. **Reconstruction error**: Measures how well a model can recreate the original data from compressed representation.
2. **Kullback–Leibler (KL) divergence term**: Quantifies how much one probability distribution differs from a reference one.

The VAE was then trained for 50 epochs using the Adam optimiser with a learning rate of 0.0001. Visual inspection of the results show good results in the image below.

![VAE reconstruction](images/vae_recon_example.png)

### UNet training

The UNet model was trained to operate on the \(4 \times 4 \times 4\) spatial latent representation produced by the VAE. It uses an encoder-decoder architecture with skip connections, progressively downsampling and upsampling spatial features. Convolutional layers are modulated by sinusoidal time embeddings \(\mathbf{t}_{emb}\) that condition the model on diffusion timesteps. CLIP text embeddings \(\mathbf{c}\) are incorporated for multimodal guidance, enabling the model to steer generation based on text prompts.

The diffusion process employs a linear beta schedule \(\beta_t\) increasing from \(\beta_{\text{start}}\) to \(\beta_{\text{end}}\) over \(T\) timesteps:

\[
\beta_t = \text{linear}(\beta_{\text{start}}, \beta_{\text{end}}, t)
\]

The forward diffusion adds noise to the original latent \(\mathbf{x}_0\) at timestep \(t\) as:

\[
\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, I)
\]

where \(\alpha_t = 1 - \beta_t\) and \(\bar{\alpha}_t = \prod_{s=1}^t \alpha_s\).

The UNet model is trained to predict the noise \(\boldsymbol{\epsilon}\) given the noisy latent \(\mathbf{x}_t\), timestep embedding \(\mathbf{t}_{emb}\), and text conditioning \(\mathbf{c}\), optimising the mean squared error loss:

\[
\mathcal{L} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c}) \right\|^2
\]

Training ran for 15 epochs using the Adam optimiser with a learning rate of 0.0002. The results of a test prompt "A cartoon face with facial hair and orange hair" are shown below.


![UNet denoising](images/example.png)

## Inference

During inference, a text prompt is first encoded into a text embedding using CLIP’s text encoder. A random noise latent tensor (matching the VAE latent space shape) is then progressively denoised by the UNet model over many diffusion steps, conditioned on the CLIP text embedding and the current timestep (via time embeddings). The UNet predicts the noise to remove at each step, gradually refining the latent towards a clean image representation. Finally, the VAE decoder transforms the denoised latent back into a full-resolution RGB image.

