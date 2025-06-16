import torch 
from transformers import CLIPProcessor, CLIPModel
from unet_architecture import UNetWithTimeEmbedding
from vae_architecture import SpatialVAE
import matplotlib.pyplot as plt
import os

def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    """
    Linear schedule for betas.
    """
    return torch.linspace(beta_start, beta_end, timesteps)

def forward_diffusion_process(x0, t, beta_schedule):
    beta = beta_schedule.to(x0.device)
    alpha = 1 - beta
    alpha_cumprod = torch.cumprod(alpha, dim=0)

    # Debug prints
    a_cumprod_t = alpha_cumprod[t].view(-1, 1, 1, 1)
    a_cumprod_t = torch.clamp(a_cumprod_t, min=1e-8, max=1.0)
    a_sqrts = torch.sqrt(a_cumprod_t)

    one_minus_cumprod_t = 1 - a_cumprod_t
    one_minus_cumprod_t = torch.clamp(one_minus_cumprod_t, min=1e-8, max=1.0)
    one_minus_sqrts = torch.sqrt(one_minus_cumprod_t)

    epsilon = torch.randn_like(x0)

    noisy = a_sqrts * x0 + one_minus_sqrts * epsilon
    return noisy, epsilon

@torch.no_grad()
def generate_image_from_text(prompt, model, vae, clip_model, clip_processor,
                             timesteps, beta_schedule, latent_channels,
                             height, width, device,
                             view_steps=[999, 750, 500, 250, 0],
                             batch_size=1):

    model.eval()
    vae.eval()
    clip_model.eval()

    beta_schedule = beta_schedule.to(device)
    alpha = 1 - beta_schedule
    alpha_cumprod = torch.cumprod(alpha, dim=0)
    alpha_cumprod_prev = torch.cat([torch.ones((1,), device=device), alpha_cumprod[:-1]], dim=0)

    # Process the prompt into text embeddings once
    inputs = clip_processor(text=[prompt], return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        text_outputs = clip_model.text_model(**inputs)
        text_features = text_outputs.pooler_output  # shape: (1, hidden_dim)
    
    # Start from pure noise at the last timestep
    x_t = torch.randn((batch_size, latent_channels, height, width), device=device)

    imgs = {}

    for t in reversed(range(timesteps)):
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

        # Predict noise at current timestep using the model (UNet), conditioning on text embeddings
        noise_pred = model(x_t, t_tensor, text=text_features)

        alpha_t = alpha[t]
        alpha_cumprod_t = alpha_cumprod[t]
        alpha_cumprod_prev_t = alpha_cumprod_prev[t]
        beta_t = beta_schedule[t]

        coef1 = 1 / torch.sqrt(alpha_t)
        coef2 = beta_t / torch.sqrt(1 - alpha_cumprod_t)
        mean = coef1 * (x_t - coef2 * noise_pred)

        if t > 0:
            noise = torch.randn_like(x_t)
            sigma_t = torch.sqrt(beta_t)
            x_t = mean + sigma_t * noise
        else:
            x_t = mean

        # Save reconstructed images at the requested timesteps
        if t in view_steps:
            with torch.no_grad():
                reconstructed = vae.decoder(x_t).clamp(0, 1).cpu()[0].permute(1, 2, 0).numpy()
            imgs[t] = reconstructed

    # Plot snapshots
    num_steps = len(view_steps)
    fig, axs = plt.subplots(1, num_steps, figsize=(5 * num_steps, 5))
    if num_steps == 1:
        axs = [axs]

    for i, t in enumerate(view_steps):
        axs[i].imshow(imgs[t])
        axs[i].set_title(f"Step {t}")
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()

import torch
import matplotlib.pyplot as plt

@torch.no_grad()
def generate_4_images_from_prompts(model, vae, clip_model, clip_processor,
                                  timesteps, beta_schedule, latent_channels,
                                  height, width, device):
    model.eval()
    vae.eval()
    clip_model.eval()

    beta_schedule = beta_schedule.to(device)
    alpha = 1 - beta_schedule
    alpha_cumprod = torch.cumprod(alpha, dim=0)
    alpha_cumprod_prev = torch.cat([torch.ones((1,), device=device), alpha_cumprod[:-1]], dim=0)

    prompts = [
        "Cartoon face with grey hair with white skin",
        "Cartoon face with orange hair and facial hair",
        "Cartoon with brown hair and brown skin",
        "Cartoon with blonde hair and yellow skin with glasses"
    ]
    batch_size = len(prompts)

    # Encode all prompts at once (batch)
    inputs = clip_processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        text_outputs = clip_model.text_model(**inputs)
        text_features = text_outputs.pooler_output  # shape: (batch_size, hidden_dim)

    # Start from pure noise for all batch
    x_t = torch.randn((batch_size, latent_channels, height, width), device=device)

    for t in reversed(range(timesteps)):
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

        noise_pred = model(x_t, t_tensor, text=text_features)

        alpha_t = alpha[t]
        alpha_cumprod_t = alpha_cumprod[t]
        beta_t = beta_schedule[t]

        coef1 = 1 / torch.sqrt(alpha_t)
        coef2 = beta_t / torch.sqrt(1 - alpha_cumprod_t)
        mean = coef1 * (x_t - coef2 * noise_pred)

        if t > 0:
            noise = torch.randn_like(x_t)
            sigma_t = torch.sqrt(beta_t)
            x_t = mean + sigma_t * noise
        else:
            x_t = mean

    # Decode final latent to images
    decoded_images = vae.decoder(x_t).clamp(0, 1).cpu()  # shape: (batch_size, C, H, W)

    # Plot images with their prompts as titles
    fig, axs = plt.subplots(1, batch_size, figsize=(4 * batch_size, 4))
    if batch_size == 1:
        axs = [axs]
    for i in range(batch_size):
        img = decoded_images[i].permute(1, 2, 0).numpy()
        axs[i].imshow(img)
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet = UNetWithTimeEmbedding(latent_channels=4)
unet.to(device)
unet.load_state_dict(torch.load(os.path.join("models", "unet_cond.pth"), map_location=device))
unet.eval()

vae = SpatialVAE()
vae.to(device)
vae.load_state_dict(torch.load(os.path.join("models", "vae.pth"), map_location=device))
vae.eval()

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.to(device)

beta_schedule = linear_beta_schedule(timesteps=1000)

# inference
# prompt = "Cartoon face with facial hair, orange hair and glasses"
# generate_image_from_text(prompt, unet, vae, clip_model, clip_processor,
#                          timesteps=1000, beta_schedule=beta_schedule,
#                          latent_channels=4, height=4, width=4, device=device)


generate_4_images_from_prompts(unet, vae, clip_model, clip_processor,
                             timesteps=1000, beta_schedule=beta_schedule, latent_channels=4,
                             height=4, width=4, device=device)