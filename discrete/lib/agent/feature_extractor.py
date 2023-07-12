import torch
from torch import nn as nn


class Residual(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, input):
        output = self.inner(input)
        return input + output


class FeatureExtractor(nn.Module):
    """
    A basic feature extractor designed to work on stacked atari frames
    Heavily based on architecture from DeepSynth and AlphaGo
    """

    def __init__(self, input_shape):
        super().__init__()

        num_blocks = 3
        num_intermediate_filters = 32
        kernel_size = (3, 3)
        padding_amount = 1

        num_channels, *input_shape_single = input_shape

        grid_size = 1
        for dim in input_shape_single:
            grid_size *= dim

        # Basically the architecture from AlphaGo
        def generate_common():
            init_conv = nn.Sequential(
                nn.Conv2d(num_channels, num_intermediate_filters, kernel_size=kernel_size, padding=padding_amount),
                nn.BatchNorm2d(num_intermediate_filters),
                nn.LeakyReLU()
            )

            blocks = [nn.Sequential(
                Residual(
                    nn.Sequential(
                        nn.Conv2d(in_channels=num_intermediate_filters,
                                  out_channels=num_intermediate_filters,
                                  kernel_size=kernel_size,
                                  padding=padding_amount),
                        nn.BatchNorm2d(num_intermediate_filters),
                        nn.LeakyReLU(),
                        nn.Conv2d(in_channels=num_intermediate_filters,
                                  out_channels=num_intermediate_filters,
                                  kernel_size=kernel_size,
                                  padding=padding_amount),
                        nn.BatchNorm2d(num_intermediate_filters))
                ),
                nn.LeakyReLU()
            ) for _ in range(num_blocks)]

            return nn.Sequential(
                init_conv, *blocks
            )

        self.net = generate_common()
        self.flattener = nn.Flatten()

        test_zeros = torch.zeros((1, *input_shape))
        self.output_size = int(self.net(test_zeros).numel())

    def forward(self, input):
        all_features = self.net(input)
        return self.flattener(all_features)

    def clone(self):
        other_featext = FeatureExtractor(self.input_shape).to(self.config.device)
        other_featext.load_state_dict(self.state_dict())
        return other_featext
