import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from generator import Generator
from perceptual_loss import PerceptualLoss


class Model:

    def __init__(self, image_size, batch_size, device, num_steps):

        K = 10  # mask parameter
        G = Generator(image_size, batch_size, K)

        self.G = G.to(device).train()
        self.optimizer = optim.Adam(self.G.parameters(), lr=5e-5, betas=(0.9, 0.999))

        def lambda_rule(i):
            decay = num_steps // 4
            m = 1.0 if i < decay else 1.0 - (i - decay) / (num_steps - decay)
            return max(m, 1e-3)

        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda_rule)
        self.perceptual_loss = PerceptualLoss().to(device)

    def train_step(self, images, T):
        """
        The input tensor represents RGB images
        with pixel values in the [0, 1] range.

        Arguments:
            images: a float tensor with shape [b, 3, h, w].
            T: an integer.
        Returns:
            a dict with float numbers.
        """
        restored_images, _, _ = self.G(images, T=T)

        losses = self.perceptual_loss(restored_images, images)
        content_loss = losses['content_loss']
        style_loss = sum(losses[n] for n in losses if 'style_loss' in n)
        perceptual_loss = content_loss + 100.0 * style_loss

        self.optimizer.zero_grad()
        perceptual_loss.backward()
        self.optimizer.step()

        # decay the learning rate
        self.scheduler.step()

        loss_dict = {
            'perceptual_loss': perceptual_loss.item(),
            'style_loss': style_loss.item()
        }

        # add all perceptual losses
        loss_dict.update(losses)

        return loss_dict

    def save_model(self, model_path):
        torch.save(self.G.state_dict(), model_path + '_generator.pth')
