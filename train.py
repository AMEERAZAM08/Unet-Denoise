import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import vgg16
import wandb

# Initialize wandb
wandb.init(project="denoise-poc")

# Function to load a single image from a local file
def load_image(image_path, image_size):
    image = Image.open(image_path).convert('RGB')  # Convert to RGB
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    image = transform(image)
    return image

# Define the dataset
class WhiteNoiseDataset(torch.utils.data.Dataset):
    def __init__(self, image, num_samples, image_size):
        self.image = image
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        noise = np.random.normal(0, 0.1, (3, self.image_size, self.image_size)).astype(np.float32)
        noisy_image = self.image + torch.tensor(noise)
        return noisy_image, self.image

# Define the neural network
class DenoiseNet(nn.Module):
    def __init__(self):
        super(DenoiseNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Define the perceptual loss
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(pretrained=True).features
        self.vgg_layers = nn.Sequential(*list(vgg)[:16]).eval()
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x_vgg = self.vgg_layers(x)
        y_vgg = self.vgg_layers(y)
        return nn.functional.mse_loss(x_vgg, y_vgg)

# Set parameters
image_path = './0.png'  # Replace with the path to your image
image_size = 64
num_samples = 1000
batch_size = 64
num_epochs = 10
learning_rate = 0.001

# Load the image and create the dataset
image = load_image(image_path, image_size)
print(type(image))
dataset = WhiteNoiseDataset(image, num_samples, image_size)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the model, loss function, and optimizer
model = DenoiseNet()
perceptual_loss = PerceptualLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Log hyperparameters to wandb
wandb.config = {
    "num_samples": num_samples,
    "image_size": image_size,
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "learning_rate": learning_rate
}

# Training loop
for epoch in range(num_epochs):
    epoch_loss = 0
    for batch_idx, (noisy_images, clean_images) in enumerate(dataloader):
        # Forward pass
        outputs = model(noisy_images)
        loss = perceptual_loss(outputs, clean_images)
        epoch_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log the loss and images to wandb every 100 batches
        if batch_idx % 100 == 0:
            wandb.log({
                "batch_loss": loss.item(),
                "epoch": epoch + 1,
                "batch_idx": batch_idx,
                "noisy_image": [wandb.Image(noisy_images[0].detach().cpu().numpy().transpose(1, 2, 0), caption="Noisy Image")],
                "denoised_image": [wandb.Image(outputs[0].detach().cpu().numpy().transpose(1, 2, 0), caption="Denoised Image")],
                "clean_images": [wandb.Image(clean_images[0].detach().cpu().numpy().transpose(1, 2, 0), caption="clean Image")]
            })

    avg_loss = epoch_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    # Log the average loss for the epoch to wandb
    wandb.log({"epoch": epoch + 1, "avg_loss": avg_loss})

# Test the model on a new noisy image
with torch.no_grad():
    noise = np.random.normal(0, 0.1, (3, image_size, image_size)).astype(np.float32)
    noisy_test_image = torch.tensor(image.numpy() + noise).unsqueeze(0)
    denoised_image = model(noisy_test_image).squeeze().cpu().numpy().transpose(1, 2, 0)

    # Log the noisy and denoised images to wandb
    wandb.log({
        "test_noisy_image": [wandb.Image(noisy_test_image.squeeze().cpu().numpy().transpose(1, 2, 0), caption="Test Noisy Image")],
        "test_denoised_image": [wandb.Image(denoised_image, caption="Test Denoised Image")]
    })


wandb.finish()
