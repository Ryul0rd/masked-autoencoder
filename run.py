import jax
import jax.numpy as jnp
from jax.nn import relu, sigmoid
import numpy as np
import haiku as hk
import optax
import chex
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import wandb

# We need these functions to get our PyTorch DataLoaders to give us numpy arrays
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

def to_numpy(pic):
    return np.array(pic, dtype=jnp.float32)

class TransformerBlock(hk.Module):
    def __init__(self, model_size, num_heads, name=None):
        super().__init__(name=name)
        self.mha = hk.MultiHeadAttention(num_heads=num_heads, key_size=model_size, w_init_scale=1.0, model_size=model_size)
        self.mlp = hk.Sequential([
            hk.Linear(model_size), relu,
            hk.Linear(model_size), relu,
        ])

    def __call__(self, x):
        x = self.mha(x, x, x) + x
        x = self.mlp(x) + x
        return x


class MaskedAutoencoder(hk.Module):
    def __init__(self, image_resolution=(28, 28), patch_resolution=(7, 7), model_size=256, encoder_blocks=2, decoder_blocks=2, mask_amount=0.75, name=None):
        super().__init__(name=name)
        self.patch_shape = image_resolution[0] // patch_resolution[0], image_resolution[1] // patch_resolution[1]
        self.num_patches = self.patch_shape[0] * self.patch_shape[1]
        self.model_size = model_size
        self.mask_amount = mask_amount

        self.patch = hk.Sequential([
            hk.Conv2D(output_channels=model_size, kernel_shape=patch_resolution, stride=patch_resolution, padding='VALID'),
            hk.Reshape(output_shape=(self.num_patches, model_size)),
        ])
        self.pos_embedding = hk.Embed(vocab_size=self.num_patches, embed_dim=model_size)
        self.encoder = hk.Sequential([TransformerBlock(model_size=model_size, num_heads=2) for _ in range(encoder_blocks)])
        self.decoder = hk.Sequential([TransformerBlock(model_size=model_size, num_heads=2) for _ in range(decoder_blocks)])
        self.unpatch = hk.Sequential([
            hk.Reshape(output_shape=(self.patch_shape[0], self.patch_shape[1], model_size)),
            hk.Conv2DTranspose(output_channels=1, kernel_shape=patch_resolution, stride=patch_resolution, padding='VALID'), sigmoid,
        ])

    def __call__(self, x):
        perm = jax.random.permutation(hk.next_rng_key(), self.num_patches)
        inv_perm = jnp.argsort(perm)
        masked_patches = int(self.num_patches * self.mask_amount)

        x = self.patch(x)
        x += self.pos_embedding(jnp.arange(self.num_patches))
        x = x[:, perm] # Permute sequence
        x = x[:, masked_patches:] # Chop off the good bit
        x = self.encoder(x)
        mask_embedding = hk.get_parameter('mask_embedding', shape=[self.model_size], dtype=x.dtype, init=jnp.zeros)
        mask_embedding = jnp.full((x.shape[0], masked_patches, self.model_size), mask_embedding)
        x = jnp.concatenate([mask_embedding, x], axis=1) # Attach 'blank'/mask embeddings
        x = x[:, inv_perm] # Reverse permutation
        x += self.pos_embedding(jnp.arange(self.num_patches))
        x = self.decoder(x)
        x = self.unpatch(x)
        return x

def main():

    # Hyperparameters
    config = {
        'learning_rate': 3e-4,
        'epochs': 3,
        'batch_size': 32,
        'rng_seed': 42,
        'log_every': 100,
        'log_images': 8,
        }

    # Do some wandb logging
    wandb.init(project='masked-autoencoder', config=config)

    # Create dataloaders
    transform = transforms.Compose([
        transforms.ToTensor(), # converts our values to be between 0. and 1. for us
        transforms.Lambda(to_numpy),
        ])
    train_ds = MNIST('/tmp/mnist/', train=True, download=True, transform=transform)
    test_ds = MNIST('/tmp/mnist/', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=wandb.config.batch_size, num_workers=0, collate_fn=numpy_collate, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=len(test_ds), num_workers=0, collate_fn=numpy_collate, drop_last=True)

    # Create our rng key
    key = jax.random.PRNGKey(wandb.config.rng_seed)

    # Define a model fn then transform it to be functionally pure
    model = hk.transform(lambda x: MaskedAutoencoder()(x))
    # Initialize some paramters using our rng keys and a tracer value
    key, subkey = jax.random.split(key)
    params = model.init(subkey, jnp.zeros(shape=(wandb.config.batch_size, 28, 28, 1)))
    # Create and init optimizer
    opt = optax.adam(wandb.config.learning_rate)
    opt_state = opt.init(params)

    # Define our loss function
    @jax.jit
    def l2_loss(params, rng_key, x):
        recon = model.apply(params, rng_key, x)
        loss = jnp.mean(optax.l2_loss(recon, x))
        return loss

    # Define our update function so we can jit it
    @jax.jit
    def update(params, opt_state, rng_key, x, y):
        loss, grads = jax.value_and_grad(l2_loss)(params, rng_key, x)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # Image logger
    def get_images(params, rng_key, x):
        originals = jnp.concatenate(x, axis=1)
        reconstructed = jnp.concatenate(model.apply(params, rng_key, x), axis=1)
        combined = jnp.concatenate([originals, reconstructed], axis=0)
        transposed = jnp.transpose(combined, (2, 0, 1))
        return wandb.Image(np.array(transposed) * 255.0)

    # Training loop
    step = 0
    for epoch in range(wandb.config.epochs):
        print(f'Starting epoch {epoch + 1}/{wandb.config.epochs}')
        for x, y in train_loader:
            x = jnp.transpose(x, (0, 2, 3, 1))
            key, subkey = jax.random.split(key)
            params, opt_state, loss = update(params, opt_state, subkey, x, y)
            if step % wandb.config.log_every == 0:
                key, subkey = jax.random.split(key)
                wandb.log({'Loss': loss, 'Samples': get_images(params, subkey, x[:wandb.config.log_images])})
            step += 1

if __name__ == '__main__':
    main()