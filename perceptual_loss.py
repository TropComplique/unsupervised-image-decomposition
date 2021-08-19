import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceptualLoss(nn.Module):

    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg = Extractor()

    def forward(self, x, y):
        """
        Arguments:
            x: a float tensor with shape [b, 3, h, w].
            y: a float tensor with shape [b, 3, h, w].
        Returns:
            two float tensors with shape [].
        """

        x = self.vgg(x)
        y = self.vgg(y)

        content_loss = []
        style_loss = []

        for n in ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']:

            content_loss.append(F.mse_loss(x[n], y[n]))
            style_loss.append(F.mse_loss(gram_matrix(x[n]), gram_matrix(y[n])))

        return sum(content_loss), sum(style_loss)


def gram_matrix(x):
    """
    Arguments:
        x: a float tensor with shape [b, c, h, w].
    Returns:
        a float tensor with shape [b, c, c].
    """
    b, c, h, w = x.shape
    x = x.view(b, c, h * w)
    g = torch.matmul(x, x.permute(0, 2, 1))
    return g.div(c * h * w)


class Extractor(nn.Module):

    def __init__(self):
        super(Extractor, self).__init__()

        from torchvision.models import vgg16
        model = vgg16(pretrained=True).features
        self.model = model.requires_grad_(False).eval()

        mean = torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

        names = []
        i, j = 1, 1

        for m in self.model:

            if isinstance(m, nn.Conv2d):
                names.append(f'conv{i}_{j}')

            elif isinstance(m, nn.ReLU):
                names.append(f'relu{i}_{j}')
                m.inplace = False
                j += 1

            elif isinstance(m, nn.MaxPool2d):
                names.append(f'pool{i}')
                i += 1
                j = 1

        # names of all features
        self.names = names

        # names of features to extract
        self.layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']

    def forward(self, x):
        """
        The input represents RGB images with
        pixel values in the [0, 1] range.

        Arguments:
            x: a float tensor with shape [b, 3, h, w].
        Returns:
            a dict with float tensors.
        """

        features = {}
        x = (x - self.mean)/self.std

        i = 0  # number of features extracted
        num_features = len(self.layers)

        for n, m in zip(self.names, self.model):
            x = m(x)

            if n in self.layers:
                features[n] = x
                i += 1

            if i == num_features:
                break

        return features
