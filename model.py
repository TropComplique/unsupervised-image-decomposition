import torch
import copy
import torch.nn.init as init
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.autograd import grad


from generator import Generator
from discriminator import MultiScaleDiscriminator
from perceptual_loss import PerceptualLoss


class Model:

    def __init__(self, device, num_steps):

        # mask parameter
        K = 10

        image_size = (128, 128)
        batch_size = 8
        K = 10
        G = Generator(image_size, batch_size, K)
        D = MultiScaleDiscriminator(3, depth=64, downsample=3, num_networks=2)

        self.G = G.to(device)
        self.D = D.to(device)

        self.optimizer = {
            'G': optim.Adam(self.G.parameters(), lr=2e-4, betas=(0.5, 0.999)),
            'D': optim.Adam(self.D.parameters(), lr=2e-4, betas=(0.5, 0.999)),
        }

        def lambda_rule(i):
            decay = num_steps // 2
            m = 1.0 if i < decay else 1.0 - (i - decay) / decay
            return max(m, 0.0)

        self.schedulers = []
        for o in self.optimizer.values():
            self.schedulers.append(LambdaLR(o, lr_lambda=lambda_rule))

        self.perceptual_loss = PerceptualLoss().to(device)

    def train_step(self, images):
        """
        The input tensor represents RGB images
        with pixel values in the [0, 1] range.

        Arguments:
            images: a float tensor with shape [b, 3, h, w].
        Returns:
            a dict with float numbers.
        """
        restored_images, _, _ = self.G(images, T=10)

        # UPDATE DISCRIMINATOR
        images.requires_grad = True
        real_scores = self.D(images)
        fake_scores = self.D(restored_images.detach())
        # they are tuples with float tensors that
        # have shape like [b, 1, some height, some width]

        real_loss = sum(F.softplus(-x).mean() for x in real_scores)
        fake_loss = sum(F.softplus(x).mean() for x in fake_scores)

        """
        Notes:
        1. softplus(x) = -log(sigmoid(-x))
        2. 1 - sigmoid(x) = sigmoid(-x)
        """

        #images.requires_grad_(True)
        
        g1 = grad(real_scores[0].sum(), images, create_graph=True)[0]
        g2 = grad(real_scores[1].sum(), images, create_graph=True)[0]
        g = g1 + g2
        # it has shape [b, 3, h, w]
        b = images.size(0)
        R1 = 0.5 * (g.view(b, -1).norm(p=2, dim=1) ** 2).mean(0)
        discriminator_loss = real_loss + fake_loss + 10 * R1

        self.optimizer['D'].zero_grad()
        discriminator_loss.backward(retain_graph=True)
        self.optimizer['D'].step()

        # UPDATE GENERATOR

        fake_scores = self.D(restored_images)
        gan_loss = sum(F.softplus(-x).mean() for x in fake_scores)
        # this is non saturating gan loss

        perceptual_loss = self.perceptual_loss(restored_images, images)
        generator_loss = perceptual_loss + gan_loss

        self.D.requires_grad_(False)
        self.optimizer['G'].zero_grad()
        generator_loss.backward()
        self.optimizer['G'].step()
        self.D.requires_grad_(True)

        # decay the learning rate
        for s in self.schedulers:
            s.step()

        loss_dict = {
            'real_loss': real_loss.item(),
            'fake_loss': fake_loss.item(),
            'R1': R1.item(),
            'discriminators_loss': discriminator_loss.item(),
            'gan_loss': gan_loss.item(),
            'perceptual_loss': perceptual_loss.item(),
            'generator_loss': generator_loss.item()
        }
        return loss_dict

    def save_model(self, model_path):
        torch.save(self.G.state_dict(), model_path + '_generator.pth')
        torch.save(self.D.state_dict(), model_path + '_discriminator.pth')
