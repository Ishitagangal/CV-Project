import torch.nn as nn
import torch.nn.functional as F
import torch

class Network(nn.Module):
    def __init__(self, classes=10):
        super(Network, self).__init__()

        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = 3
        for out_filters, stride, normalize in [(64, 2, False), (128, 2, True), (256, 2, True), (512, 1, True)]:
            layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, 3, 1, 1)) 
        layers.append(Flatten())

        layers.append(nn.Linear(256,100))
        layers.append(nn.Linear(100,classes))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)