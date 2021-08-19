import torch
import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, w=128, h=128, K=29):
        super(Generator, self).__init__()

        self.resnet = Resnet(K)
        self.mask = MaskGenerator(w, h, K)

        def weights_init(m):
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(weights_init)

    def forward(self, images, T=10):
        """
        The input tensor represents RGB images
        with pixel values in the [0, 1] range.

        Arguments:
            images: a float tensor with shape [b, 3, h, w].
            T: an integer, the number of masks and colors.
        Returns:
            results: a list of float tensors with shape [b, 3, h, w], restored images.
            masks: a list of float tensors with shape [b, 1, h, w].
            colors: a list of float tensors with shape [b, 3, 1, 1].
        """

        results = []
        masks = []
        colors = []

        x = torch.zeros_like(images)
        # it has shape [b, 3, h, w] and
        # represents an initial black canvas

        for _ in range(T):

            p, c = self.resnet(torch.cat([x.detach(), images], dim=1))
            # they have shapes [b, K] and [b, 3]

            m = self.mask(p)
            # it has shape [b, 1, h, w]

            c = c.unsqueeze(2).unsqueeze(2)
            # it has shape [b, 3, 1, 1]

            x = x * (1.0 - m) + c * m
            # it has shape [b, 3, h, w]

            results.append(x)
            masks.append(m.detach())
            colors.append(c.detach())

        return results, masks, colors


class MaskGenerator(nn.Module):

    def __init__(self, w, h, K, depth=128):
        super(MaskGenerator, self).__init__()

        self.layers = nn.Sequential(

            nn.GroupNorm(num_groups=32, num_channels=K + 2),
            nn.Conv2d(K + 2, depth, kernel_size=1),
            nn.Tanh(),

            nn.GroupNorm(num_groups=32, num_channels=depth),
            nn.Conv2d(depth, depth, kernel_size=1),
            nn.Tanh(),

            nn.GroupNorm(num_groups=32, num_channels=depth),
            nn.Conv2d(depth, depth, kernel_size=1),
            nn.Tanh(),

            nn.Conv2d(depth, 1, kernel_size=1),
            nn.Sigmoid()
        )

        coordinates = generate_coordinates(w, h)
        self.register_buffer('coordinates', coordinates)
        # it has shape [1, 2, h, w]

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, K].
        Returns:
            a float tensor with shape [b, 1, h, w].
        """

        b, K = x.shape
        h, w = self.coordinates.shape[2:]

        x = x.view(b, K, 1, 1).expand(-1, -1, h, w)
        c = self.coordinates.expand(b, -1, -1, -1)

        x = torch.cat([x, c], dim=1)
        return = self.layers(x)


class Resnet(nn.Module):

    def __init__(self, K):
        """
        Arguments:
            K: an integer.
        """
        super(Resnet, self).__init__()

        from torchvision.models import resnet18
        model = resnet18(pretrained=False, num_classes=K + 3)
        model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.K = K
        self.model = model

    def forward(self, x):
        """
        The input tensor represents a batch of RGB image
        pairs with pixel values in the [0, 1] range.

        Arguments:
            x: a float tensor with shape [b, 6, h, w].
        Returns:
            p: a float tensor with shape [b, K].
            c: a float tensor with shape [b, 3].
        """
        x = 2.0 * x - 1.0
        x = self.model(x)

        p, c = torch.split(x, [self.K, 3], dim=1)
        # they have shapes [b, K] and [b, 3]

        c = torch.sigmoid(c)
        return p, c


def generate_coordinates(w, h):
    """
    Arguments:
        w, h: integers.
    Returns:
        a float tensor with shape [1, 2, h, w].
    """

    y, x = torch.meshgrid(
        torch.arange(0, h, dtype=torch.float32),
        torch.arange(0, w, dtype=torch.float32)
    )
    c = torch.stack([y, x], dim=0)

    # it is true that c[:, a, b] = [a, b]
    # for all indices a and b

    scaler = torch.FloatTensor([h - 1, w - 1])
    c /= scaler.view(2, 1, 1)  # to the [0, 1] range

    return c.unsqueeze(0)
