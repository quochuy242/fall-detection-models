import torch
import torch.nn as nn
from typing import List
import numpy as np
from config import CNNArgument, ViTArgument


class CNN(nn.Module):
    def __init__(self, arguments: CNNArgument):
        super(CNN, self).__init__()

        # Define the convolutional layers as a block
        self.features = nn.Sequential(
            # Block 1
            self._conv_block(
                1,
                arguments.conv_filter_size[0],
                arguments.conv_kernel_size[0],
                arguments.conv_padding,
                arguments.pool_kernel_size,
                arguments.pool_stride,
            ),
            # Block 2
            self._conv_block(
                arguments.conv_filter_size[0],
                arguments.conv_filter_size[-1],
                arguments.conv_kernel_size[-1],
                arguments.conv_padding,
                arguments.pool_kernel_size,
                arguments.pool_stride,
            ),
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                arguments.conv_filter_size[-1] * np.prod(arguments.conv_kernel_size),
                arguments.fc_units[0],
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(p=arguments.dropout_rate),
            nn.Linear(*arguments.fc_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=arguments.dropout_rate),
            nn.Linear(arguments.fc_units[-1], arguments.num_classes),
        )

    def _conv_block(
        conv1: int,
        conv2: int,
        kernel_size: int = 3,
        padding: int = 1,
        pool_size: int = 2,
        pool_stride: int = 2,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(conv1, conv2, kernel_size, padding),
            nn.BatchNorm2d(conv1),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv2, conv2, kernel_size, padding),
            nn.BatchNorm2d(conv2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool_size, pool_stride),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# TODO: Write ViT model class here
class ViT(nn.Module): ...
