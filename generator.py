import torch
import torch.nn as nn
from torchvision.models import resnet18


class Generator(nn.Module):

    def __init__(self, image_size, batch_size, K):
        """
        Arguments:
            image_size: a tuple of integers (width, height).
            batch_size: an integer.
            K: an integer.
        """
        super(Generator, self).__init__()

        self.resnet = Resnet(K)

        # mask generator
        self.f = nn.Sequential(
            nn.Linear(K + 2, 128),
            nn.GroupNorm(32, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.GroupNorm(32, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.GroupNorm(32, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        b = batch_size
        w, h = image_size

        coordinates = generate_coordinates(w, h)
        coordinates = coordinates.unsqueeze(0).to(device)
        self.coordinates = coordinates.repeat(b, 1, 1, 1)
        # it has shape [b, h, w, 2]

        def weights_init(m):
            if isinstance(m, nn.GroupNorm):
                init.ones_(m.weight)
                init.zeros_(m.bias)
            else:
                assert isinstance(m, nn.Linear)
                init.ones_(m.weight)
                init.zeros_(m.bias)

        self.f = self.f.apply(weights_init)

    def forward(self, images, T=10):
        """
        The input tensor represents RGB images
        with pixel values in the [0, 1] range.

        Arguments:
            images: a float tensor with shape [b, 3, h, w].
            T: an integer, the number of masks and colors.
        Returns:
            x: a float tensor with shape [b, 3, h, w], restored images.
            masks: a list of float tensors with shape [b, 1, h, w].
            colors: a list of float tensors with shape [b, 3, 1, 1].
        """
        masks = []
        colors = []

        x = torch.ones_like(images)
        # it has shape [b, 3, h, w] and
        # represents an initial white canvas

        for _ in range(T):

            z = torch.cat([x, images], dim=1)
            p, c = self.resnet(z)
            # they have shapes [b, K] and [b, 3]

            p = p.unsqueeze(1).unsqueeze(1)
            p = p.repeat(1, h, w, 1)
            # it has shape [b, h, w, K]

            y = torch.cat([self.coordinates, p], dim=3)
            # it has shape [b, h, w, K + 2]

            K = p.size(1)
            y = y.reshape(-1, K + 2)
            M = self.f(y)  # shape [b * h * w, 1]
            M = M.reshape(b, h, w, 1)
            M = M.permute(0, 3, 1, 2)
            # it has shape [b, 1, h, w]

            c = c.unsqueeze(2).unsqueeze(2)
            # it has shape [b, 3, 1, 1]

            x = x * (1.0 - M) + c * M
            # it has shape [b, 3, h, w]

            masks.append(M)
            colors.append(c)

        return x, masks, colors


class Resnet(nn.Module):

    def __init__(self, K):
        """
        Arguments:
            K: an integer.
        """
        super(Resnet, self).__init__()

        model = resnet18(pretrained=False, num_classes=K + 3)
        model.conv1 = nn.Conv2d(6, model.inplanes, kernel_size=7, stride=2, padding=3, bias=False)

        self.K = K
        self.model = model

    def forward(self, x, T):
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
        a float tensor with shape [h, w, 2].
    """

    y, x = torch.meshgrid(
        torch.arange(0, h, dtype=torch.float32),
        torch.arange(0, w, dtype=torch.float32)
    )
    c = torch.stack([y, x], axis=2)

    # it is true that c[a, b] = [a, b]
    # for all indices a and b

    scaler = torch.FloatTensor([h - 1, w - 1])
    c /= scaler  # convert to the [0, 1] range

    return c
