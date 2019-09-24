import torch
import torch.nn as nn
import torch.nn.init as init


class MultiScaleDiscriminator(nn.Module):

    def __init__(self, in_channels, depth=64, downsample=3, num_networks=2):
        super(MultiScaleDiscriminator, self).__init__()

        networks = []
        for _ in range(num_networks):
            layers = get_layers(in_channels, depth, downsample)
            networks.append(layers)

        self.networks = nn.ModuleList(networks)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm2d):
                init.ones_(m.weight)
                init.zeros_(m.bias)

        self.apply(weights_init)

    def forward(self, x):
        """
        I assume that h and w are divisible
        by 2**(downsample + num_networks - 1).

        The input tensor represents RGB images
        with pixel values in the [0, 1] range.

        Arguments:
            x: a float tensor with shape [b, in_channels, h, w].
         Returns:
            scores: a list with float tensors.
        """
        scores = []

        num_networks = len(self.networks)
        x = 2.0 * x - 1.0

        for i, n in enumerate(self.networks):

            scores.append(n(x))

            if i != num_networks - 1:
                x = self.downsample(x)

        return scores


def get_layers(in_channels, depth, downsample):

    params = {
        'kernel_size': 4, 'stride': 2,
        'padding': 1, 'bias': True
    }
    # (note that I use 'same' padding here)

    out_channels = depth
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, **params),
        nn.LeakyReLU(0.2, inplace=True)
    )
    downsampling = [block]

    params['bias'] = False
    # bias is not needed
    # because of normalization

    for n in range(1, downsample):

        in_channels = out_channels
        out_channels = depth * min(2**n, 8)

        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **params),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        downsampling.append(block)

    """
    Right now receptive field is
    22 if downsample = 3,
    46 if downsample = 4,
    94 if downsample = 5.

    And the image size is reduced
    in `2**downsample` times.
    """

    penultimate_block = nn.Sequential(
        nn.Conv2d(out_channels, out_channels, kernel_size=4, bias=False),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.LeakyReLU(0.2, inplace=True),
    )
    final_block = nn.Conv2d(out_channels, 1, kernel_size=4)
    # (note that I use 'valid' padding here)

    """
    Right now receptive field is
    34 if downsample = 2,
    70 if downsample = 3,
    142 if downsample = 4,
    286 if downsample = 5.

    See https://fomoro.com/projects/project/receptive-field-calculator
    """
    return nn.Sequential(*(downsampling + [penultimate_block, final_block]))
