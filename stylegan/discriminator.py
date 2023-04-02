import torch
import torch.nn as nn

class StyleGANDiscriminatorBlock(nn.Module):
    '''
    StyleGAN Discriminator Block Class
    Values:
        in_chan: the number of channels in the input feature map
        out_chan: the number of channels in the output feature map
    '''

    def __init__(self, in_chan, out_chan):
        super(StyleGANDiscriminatorBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, in_chan, kernel_size=3, stride=1, padding=1, groups=in_chan)
        self.conv2 = nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=1)
        self.activation = nn.LeakyReLU(0.2)
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        '''
        Function for completing a forward pass of StyleGANDiscriminatorBlock: Given an x, 
        computes a StyleGAN discriminator block.
        '''
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.downsample(x)
        return x

class StyleGANDiscriminator(nn.Module):
    '''
    StyleGAN Discriminator Class
    Values:
        channels: a list of channel sizes for each block in the discriminator
    '''

    def __init__(self, channels=[8, 16, 32, 64, 128, 256, 512]):
        super(StyleGANDiscriminator, self).__init__()

        self.num_layers = len(channels)
        self.blocks = nn.ModuleList([])
        for i in range(self.num_layers):
            in_channels = 3 if i == 0 else channels[i-1]
            out_channels = channels[i]
            block = StyleGANDiscriminatorBlock(in_channels, out_channels)
            self.blocks.append(block)

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, 1)
        
        # LeakyReLU activation function
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        # Sigmoid activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        '''
        Function for completing a forward pass of StyleGANDiscriminator: Given an x, 
        computes a StyleGAN discriminator output.
        '''
        for i in range(self.num_layers):
            x = self.blocks[i](x)
        
        # Flatten the output and concatenate with labels
        x = x.view(-1, 512 * 2 * 2)

        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        
        # Final output layer (sigmoid activation)
        output = self.sigmoid(x)
        
        return output

if __name__ == "__main__":
    # create an instance of the discriminator
    discriminator = StyleGANDiscriminator()

    # generate some fake input data (batch_size, in_channels, height, width)
    input_data = torch.randn(8, 3, 256, 256)

    # pass the input data through the discriminator
    output = discriminator(input_data)

    # print the output shape
    print(output.shape) # should be (1,)