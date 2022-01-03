import jax
import jax.numpy as jnp
from jax.nn import relu, sigmoid
import numpy as np
import haiku as hk
import optax
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from tqdm import tqdm
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
        self.ln1 = hk.LayerNorm(-1, False, False)
        self.mha = hk.MultiHeadAttention(num_heads=num_heads, key_size=model_size, w_init_scale=1.0, model_size=model_size)
        self.ln2 = hk.LayerNorm(-1, False, False)
        self.mlp = hk.Sequential([
            hk.Linear(model_size), relu,
            hk.Linear(model_size), relu,
        ])

    def __call__(self, x):
        norm = self.ln1(x)
        x += self.mha(norm, norm, norm)
        norm = self.ln2(x)
        x += self.mlp(norm)
        return x


class MaskedAutoencoder(hk.Module):
    def __init__(self, image_shape=(32, 32, 3), patch_resolution=(4, 4), mask_amount=3/4, num_heads=2,
                 d_encoder=256, d_decoder=128, encoder_blocks=2, decoder_blocks=2, name=None):
        super().__init__(name=name)
        self.patch_shape = image_shape[0] // patch_resolution[0], image_shape[1] // patch_resolution[1]
        self.num_patches = self.patch_shape[0] * self.patch_shape[1]
        self.d_encoder = d_encoder
        self.mask_amount = mask_amount

        self.patch = hk.Sequential([
            hk.Conv2D(output_channels=d_encoder, kernel_shape=patch_resolution, stride=patch_resolution, padding='VALID'),
            hk.Reshape(output_shape=(self.num_patches, d_encoder)),
        ])
        self.encoder = hk.Sequential([TransformerBlock(model_size=d_encoder, num_heads=num_heads) for _ in range(encoder_blocks)])
        self.projection = hk.Linear(d_decoder)
        self.decoder = hk.Sequential([TransformerBlock(model_size=d_decoder, num_heads=num_heads) for _ in range(decoder_blocks)])
        self.unpatch = hk.Sequential([
            hk.Reshape(output_shape=(self.patch_shape[0], self.patch_shape[1], d_decoder)),
            hk.Conv2DTranspose(output_channels=image_shape[2], kernel_shape=patch_resolution, stride=patch_resolution, padding='VALID'), sigmoid,
        ])

    def __call__(self, x):
        perm = jax.random.permutation(hk.next_rng_key(), self.num_patches)
        inv_perm = jnp.argsort(perm)
        masked_patches = int(self.num_patches * self.mask_amount)
        pos_embedding = hk.get_parameter('pos_embedding', shape=[self.num_patches, self.d_encoder], dtype=x.dtype, init=hk.initializers.TruncatedNormal())

        x = self.patch(x)
        x += pos_embedding
        x = x[:, perm] # Permute sequence
        x = x[:, masked_patches:] # Chop off the good bit
        x = self.encoder(x)
        mask_embedding = hk.get_parameter('mask_embedding', shape=[self.d_encoder], dtype=x.dtype, init=jnp.zeros)
        mask_embedding = jnp.full((x.shape[0], masked_patches, self.d_encoder), mask_embedding)
        x = jnp.concatenate([mask_embedding, x], axis=1) # Attach 'blank'/mask embeddings
        x = x[:, inv_perm] # Reverse permutation
        x += pos_embedding
        x = self.projection(x)
        x = self.decoder(x)
        x = self.unpatch(x)
        return x

def main():
    config = {
        'learning_rate': 3e-4,
        'warmup_steps': 5000,
        'epochs': 15,
        'batch_size': 32,
        'patch_resolution': (4, 4),
        'mask_amount': 3/4,
        'num_heads': 2,
        'd_encoder': 256,
        'd_decoder': 192,
        'encoder_depth': 2,
        'decoder_depth': 2,
        'rng_seed': 42,
        'log_every': 50,
        'log_images': 8,
        }

    # Do some wandb logging
    wandb.init(project='masked-autoencoder', config=config)

    # Create dataloaders
    transform = transforms.Compose([
        transforms.ToTensor(), # converts our values to be between 0. and 1. for us
        transforms.Lambda(to_numpy),
        transforms.Lambda(lambda x: jnp.transpose(x, (1, 2, 0))),
        ])
    train_ds = CIFAR10('~/Documents/datasets/cifar10/', train=True, download=True, transform=transform)
    test_ds = CIFAR10('~/Documents/datasets/cifar10/', train=False, download=False, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=wandb.config.batch_size, num_workers=0, collate_fn=numpy_collate, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=wandb.config.batch_size, num_workers=0, collate_fn=numpy_collate, drop_last=True)

    # Create our rng key
    key = jax.random.PRNGKey(wandb.config.rng_seed)

    # Define a model fn then transform it to be functionally pure
    def f_model(x):
        nn = MaskedAutoencoder(patch_resolution=wandb.config.patch_resolution, mask_amount=wandb.config.mask_amount,
                               num_heads=wandb.config.num_heads, d_encoder=wandb.config.d_encoder, d_decoder=wandb.config.d_decoder,
                               encoder_blocks=wandb.config.encoder_depth, decoder_blocks=wandb.config.decoder_depth)
        return nn(x)
    model = hk.transform(f_model)
    # Initialize some paramters using our rng keys and a tracer value
    key, subkey = jax.random.split(key)
    params = model.init(subkey, jnp.zeros((wandb.config.batch_size, 32, 32, 3)))
    # Create and init optimizer
    lr_schedule = optax.warmup_cosine_decay_schedule(wandb.config.learning_rate / 10000, wandb.config.learning_rate, 
                                                     wandb.config.warmup_steps, wandb.config.epochs * 1562)
    opt = optax.adam(lr_schedule)
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
        return wandb.Image(np.array(combined * 255.0))

    # Training loop
    step = 0
    for epoch in range(wandb.config.epochs):
        for x, y in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{wandb.config.epochs}'):
            key, subkey = jax.random.split(key)
            params, opt_state, loss = update(params, opt_state, subkey, x, y)
            if step % wandb.config.log_every == 0:
                key, subkey = jax.random.split(key)
                wandb.log({'Loss': loss, 'Samples': get_images(params, subkey, x[:wandb.config.log_images])}, step=step)
            step += 1
        key, subkey = jax.random.split(key)
        val_loss = 0
        for x, y in test_loader:
            val_loss += l2_loss(params, subkey, x)
        val_loss /= len(test_ds) / wandb.config.batch_size
        wandb.log({'Val Loss': val_loss, 'Val Samples': get_images(params, subkey, x[:wandb.config.log_images])}, step=step)
        

if __name__ == '__main__':
    main()