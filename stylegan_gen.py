# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython
import torch
from model.Stylegan import StyleGAN2
from utils.utils import styles_def_to_tensor, noise, evaluate_in_chunks, latent_to_w, noise_list, image_noise
import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt

# %%
GAN = None
load_model_name = 'model_10.pt'
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
GAN = StyleGAN2(lr=2e-4,
                image_size=64,
                network_capacity=16,
                transparent=False,)
GAN.to(device)
GAN.load_state_dict(torch.load(load_model_name,
                               map_location=torch.device(device)))

# %%
@torch.no_grad()
def generate_truncated(self, S, G, style, noi, trunc_psi=0.6, num_image_tiles=8):
    latent_dim = G.latent_dim

    if self.av is None:
        z = noise(2000, latent_dim)
        samples = evaluate_in_chunks(self.batch_size, S, z).cpu().numpy()
        self.av = np.mean(samples, axis=0)
        self.av = np.expand_dims(self.av, axis=0)

    w_space = []
    for tensor, num_layers in style:
        tmp = S(tensor)
        av_torch = torch.from_numpy(self.av).to(device)
        tmp = trunc_psi * (tmp - av_torch) + av_torch
        w_space.append((tmp, num_layers))

    w_styles = styles_def_to_tensor(w_space)
    generated_images = evaluate_in_chunks(
        self.batch_size, G, w_styles, noi)
    return generated_images.clamp_(0., 1.)


# %%
GAN.eval()
transparent = False
num_image_tiles = 8
batch_size = 64
ext = 'jpg' if not transparent else 'png'
num_rows = num_image_tiles


def generate_images(stylizer, generator, latents, noise):
    w = latent_to_w(stylizer, latents)
    w_styles = styles_def_to_tensor(w)
    generated_images = evaluate_in_chunks(batch_size, generator,
                                          w_styles, noise)
    generated_images.clamp_(0., 1.)
    return generated_images


latent_dim = GAN.G.latent_dim
image_size = GAN.G.image_size
num_layers = GAN.G.num_layers

# latents and noise
latents = noise_list(num_rows**2, num_layers, latent_dim)
n = image_noise(num_rows**2, image_size)

# regular
generated_images = generate_images(GAN.S, GAN.G, latents, n)


# %%
plt.figure(figsize=(10, 10))
plt.axis("off")
plt.title("Ori Images")
plt.imshow(np.transpose(vutils.make_grid(generated_images,
                                         padding=2,
                                         normalize=True).detach().numpy(),
                        (1, 2, 0)))


# %%
