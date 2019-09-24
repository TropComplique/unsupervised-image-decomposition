import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16


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
            a dict with float tensors with shape [].
        """
        losses = {}

        x = self.vgg(x)
        y = self.vgg(y)

        content_loss = F.mse_loss(x['relu3_3'], y['relu3_3'])
        losses['content_loss'] = content_loss

        for n in ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']:

            gram_x = gram_matrix(x[n])
            gram_y = gram_matrix(y[n])

            style_loss = F.mse_loss(gram_x, gram_y)
            losses[f'style_loss_{n}'].append(style_loss)

        return losses


def gram_matrix(x):
    """
    Arguments:
        x: a float tensor with shape [b, c, h, w].
    Returns:
        a float tensor with shape [b, c, c].
    """
    b, c, h, w = x.size()
    x = x.view(b, c, h * w)
    g = torch.matmul(x, x.permute(0, 2, 1))
    return g.div(c * h * w)


class Extractor(nn.Module):

    def __init__(self):
        super(Extractor, self).__init__()

        model = vgg16(pretrained=True)
        self.model = model.eval().features
        self.model.requires_grad_(False)

        # normalization
        mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.mean = nn.Parameter(data=mean, requires_grad=False)
        self.std = nn.Parameter(data=std, requires_grad=False)

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
        Arguments:
            x: a float tensor with shape [b, 3, h, w].
            It represents RGB images with pixel values in [0, 1] range.
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
