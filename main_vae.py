from torch.utils.data import DataLoader
from torch.optim import Adam

from utils.loaders import load_mnist

from models.VAE import CustomDataset, VariationalAutoEncoder
from models.VAE import vae_loss

(x_train, y_train), (x_test, y_test) = load_mnist()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


BATCH_SIZE = 100

train_dataset = CustomDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

vae = VariationalAutoEncoder(in_channels=1, z_dim=111, encoder_hidden_dims=[32, 64, 128, 256, 512], batch_size=BATCH_SIZE)
optimizer = Adam(vae.parameters())

epochs = 1
for i in range(epochs):
    print("## Epoch:", i+1)
    for batch_idx, data in enumerate(train_loader):
        print(f"# Batch number:{batch_idx+1}")
        x, y = data
        # clear gradients
        optimizer.zero_grad()
        # predict
        x_pred = vae(x)
        # loss
        loss = vae_loss(x, x_pred, vae.mu_unit, vae.log_var_unit, r_loss_factor=1000)
        # back-propagation
        loss.backward()
        # update params based on gradients
        optimizer.step()
        print("Final loss:", loss.data)
        print("="*150)