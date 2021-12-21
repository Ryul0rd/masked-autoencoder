import jax
import jax.numpy as jnp
from jax.nn import relu, sigmoid
import numpy as np
import haiku as hk
import optax
import chex
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

class Autoencoder(hk.Module):

    def __init__(self, name=None):
        super().__init__(name=name)
        self.nn = hk.Sequential([
            hk.Flatten(),
            hk.Linear(256), jax.nn.relu,
            hk.Linear(128), jax.nn.relu,
            hk.Linear(256), jax.nn.relu,
            hk.Linear(784), jax.nn.sigmoid,
        ])

    def __call__(self, x):
        return jnp.reshape(self.nn(x), (-1, 1, 28, 28))


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

    def __init__(self, model_size=128, blocks=2, name=None):
        super().__init__(name=name)
        self.nn = hk.Sequential([
            hk.Conv2D(output_channels=model_size, kernel_shape=4, stride=4),
            hk.Reshape(output_shape=(49, model_size)),
            hk.Sequential([TransformerBlock(model_size=model_size, num_heads=2) for _ in range(blocks)]),
            hk.Reshape(output_shape=(7, 7, model_size)),
            hk.Conv2DTranspose(output_channels=1, kernel_shape=4, stride=4),
        ])
        # self.patch = hk.Conv2D(output_channels=model_size, kernel_shape=4, stride=4)
        # self.transformer = hk.Sequential([TransformerBlock(model_size=model_size, num_heads=2) for _ in range(2)])
        # self.unpatch = hk.Conv2DTranspose(output_channels=1, kernel_shape=4, stride=4)

    def __call__(self, x):
        # x = relu(self.patch(x))
        # x = jnp.reshape(x, (-1, 49, self.model_size))
        # x = self.transformer(x)
        # x = jnp.reshape(x, (-1, 7, 7, self.model_size))
        # x = sigmoid(self.unpatch(x))
        return self.nn(x)

def main():

    # Hyperparameters
    config = {
        'learning_rate': 3e-4,
        'epochs': 2,
        'batch_size': 32,
        'rng_seed': 42,
        'log_every': 100,
        'log_images': 8,
        }

    # Do some wandb logging
    wandb.init(project="masked-autoencoder", config=config)

    # Create dataloaders
    transform = transforms.Compose([
        transforms.ToTensor(), # converts our values to be between 0. and 1. for us
        transforms.Lambda(to_numpy),
        ])
    train_ds = MNIST('/tmp/mnist/', train=True, download=True, transform=transform)
    test_ds = MNIST('/tmp/mnist/', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=wandb.config.batch_size, num_workers=0, collate_fn=numpy_collate)
    test_loader = DataLoader(test_ds, batch_size=len(test_ds), num_workers=0, collate_fn=numpy_collate)

    # Create our rng key
    key = jax.random.PRNGKey(wandb.config.rng_seed)

    # Define a model fn then transform it to be functionally pure
    model = hk.without_apply_rng(hk.transform(lambda x: MaskedAutoencoder()(x)))
    # Initialize some paramters using our rng keys and a tracer value
    key, subkey = jax.random.split(key)
    params = model.init(subkey, jnp.zeros(shape=(wandb.config.batch_size, 28, 28, 1)))
     # Create and init optimizer
    opt = optax.adam(wandb.config.learning_rate)
    opt_state = opt.init(params)

    # Define our loss function
    @jax.jit
    def loss(params, x):
        recon = model.apply(params, x)
        l2 = jnp.mean(optax.l2_loss(recon, x))
        return l2

    # Define our update function so we can jit it
    @jax.jit
    def update(params, opt_state, x, y):
        l2, grads = jax.value_and_grad(loss)(params, x)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, l2

    # Image logger
    def get_images(params, x):
        originals = jnp.concatenate(x, axis=2)
        reconstructed = jnp.concatenate(model.apply(params, x), axis=2)
        combined = jnp.concatenate([originals, reconstructed], axis=1)
        transposed = jnp.transpose(combined, (0, 3, 1, 2))
        return wandb.Image(np.array(transposed) * 255.0)

    # Training loop
    step = 0
    for epoch in range(wandb.config.epochs):
        for x, y in train_loader:
            x = jnp.transpose(x, (0, 2, 3, 1))
            params, opt_state, l2 = update(params, opt_state, x, y)
            if step % wandb.config.log_every == 0:
                print(x.shape)
                print(model.apply(params, x).shape)
                wandb.log({'Loss': l2, 'Samples': get_images(params, x[:wandb.config.log_images])})
            step += 1

if __name__ == '__main__':
    main()