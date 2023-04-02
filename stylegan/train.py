import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from tqdm import tqdm

from generator import StyleGANGenerator
from discriminator import StyleGANDiscriminator

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the dataset and dataloader
data_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
anime_faces = ImageFolder('./anime-face-dataset', transform=data_transforms)
dataloader = DataLoader(anime_faces, batch_size = 8, shuffle = True, num_workers=4)

# Define the generator and discriminator models
generator = StyleGANGenerator(z_dim=512, map_hidden_dim=512, w_dim=512).to(device)
discriminator = StyleGANDiscriminator().to(device)

# Define the loss functions
adversarial_loss = torch.nn.BCELoss()
perceptual_loss = torch.nn.L1Loss()

# Set up the optimizer and learning rate scheduler
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001, betas=(0.0, 0.99))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001, betas=(0.0, 0.99))
g_scheduler = torch.optim.lr_scheduler.ExponentialLR(g_optimizer, gamma=0.99)
d_scheduler = torch.optim.lr_scheduler.ExponentialLR(d_optimizer, gamma=0.99)

# Define the training loop
num_epochs = 100

# Train models
cur_step = 0
display_step = 500
mean_generator_loss = 0
mean_discriminator_loss = 0

visualization = False

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    for i, (real_images, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        
        # Train the discriminator
        discriminator.zero_grad()
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Generate fake images
        z = torch.randn(batch_size, 512).to(device)
        fake_images = generator(z)
        

        # Compute discriminator loss
        real_logits = discriminator(real_images)
        fake_logits = discriminator(fake_images.detach())
        
        d_loss_real = adversarial_loss(real_logits, real_labels)
        d_loss_fake = adversarial_loss(fake_logits, fake_labels)
        d_loss = d_loss_real + d_loss_fake
        
        # Keep track of the average discriminator loss
        mean_discriminator_loss += d_loss.item() / display_step
        
        d_loss.backward()
        d_optimizer.step()
        
        # Train the generator
        generator.zero_grad()
        z = torch.randn(batch_size, 512).to(device)
        fake_images = generator(z)

        # Compute generator loss
        fake_logits = discriminator(fake_images)
        g_loss_adversarial = adversarial_loss(fake_logits, real_labels)
        g_loss_perceptual = perceptual_loss(fake_images, real_images)
        g_loss = g_loss_adversarial + 10.0 * g_loss_perceptual
        g_loss.backward()
        g_optimizer.step()

        # Update the learning rate
        g_scheduler.step()
        d_scheduler.step()
        
        # Keep track of the average generator loss
        mean_generator_loss += g_loss.item() / display_step

        ## Visualization code ##
        if cur_step % display_step == 0 and cur_step > 0:
            print(f"Epoch {epoch + 1}, Step {cur_step}: \nGenerator loss: {mean_generator_loss} | Discriminator loss: {mean_discriminator_loss}\n")

            mean_generator_loss = 0
            mean_discriminator_loss = 0
        cur_step += 1

        # Save the model weights
        if (epoch + 1) % 10 == 0:
            torch.save(generator, f"stylegan_generator_epoch{epoch+1}.pth")
            torch.save(discriminator, f"stylegan_discriminator_epoch{epoch+1}.pth")
