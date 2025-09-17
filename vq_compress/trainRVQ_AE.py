import torch
import numpy as np
import torchvision
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from vector_quantize_pytorch import ResidualVQ
from torch.nn.functional import cosine_similarity
import matplotlib.pyplot as plt
from torch.optim import Adam
from vq_init import ResidualVectorQuantize 

mse_loss = nn.MSELoss()
transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = torchvision.datasets.FashionMNIST(root='FashionMNIST/raw/t10k-images-idx3-ubyte', transform=transform, download = True)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Build AE model (Undercomplte)
class UnderComplete(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(UnderComplete, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_layer = nn.Linear(input_size, hidden_size)

        self.quantizer = ResidualVectorQuantize(
            input_dim=hidden_size,
            codebook_dim = 64,
            n_q=4,
            bins=32       
        )

        self.output_layer = nn.Linear(hidden_size, input_size)
        self.relu = nn.ReLU() # bcoz, the inputs are always positive or 0

    def forward(self, x):
        x = x.view(-1, self.input_size) # 28*28
        x = self.hidden_layer(x)
        x = self.relu(x)
        x = self.quantizer(x)[0]
        x = self.output_layer(x)
        x = self.relu(x)
        return x
  
under_completeAE = UnderComplete(28*28, 100)
under_completeAE = under_completeAE.to('cuda')


# Training
def fit_undercompleteAE(under_completeAE, epochs = 100, lr = 1e-3, eps = 1e-8):
    optimizer = Adam(under_completeAE.parameters(), lr=lr, eps = eps)
    under_completeAE = under_completeAE.to('cuda')
    for epoch in range(1, epochs+1):
        epoch_loss= 0
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            batch = batch.to('cuda')
            x, y = batch
            x = x.view(-1, 28*28)

            # fwd prop
            x_hat = under_completeAE(x)
            # loss
            loss = mse_loss(x, x_hat)
            # back prop
            loss.backward()
            # update
            optimizer.step()
            with torch.no_grad():
              epoch_loss += loss
    print(f"Loss: {epoch}", epoch_loss/len(train_dataloader))

for batch in test_dataloader:
    plt.figure(figsize=(12, 4))
    under_completeAE.eval()
    x = batch[0][10]
    plt.subplot(1, 2, 1)
    plt.imshow(x.squeeze(), cmap='grey')
    plt.title("Input Image")
    plt.show()

    # Fwd Prop
    under_completeAE = under_completeAE.to('cpu')
    x_hat = under_completeAE(x.view(-1, 28*28))
    x_hat = x_hat.view(-1, 28, 28)
    # detach from comp. graph

    x_hat = x_hat.detach()
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 2)
    plt.imshow(x_hat.squeeze(), cmap='grey')
    plt.title("Reconstructed Image")
    plt.show()
    break