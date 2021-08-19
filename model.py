import os

import torch
import torch.nn as nn
import torch.optim as optim

from generator import Generator
from perceptual_loss import PerceptualLoss
from torch.optim.lr_scheduler import LambdaLR


class Model:

    def __init__(self, device, num_steps):

        self.generator = Generator(w=128, h=128).to(device).train()
        self.optimizer = optim.Adam(self.generator.parameters(), lr=2e-4, betas=(0.9, 0.999))

        def lambda_rule(i):
            decay = num_steps // 2
            m = 1.0 if i < decay else 1.0 - (i - decay) / (num_steps - decay)
            return max(m, 1e-3)

        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda_rule)
        self.perceptual_loss = PerceptualLoss().to(device)

        self.states = {
            'generator': self.generator,
            'optimizer': self.optimizer,
            'scheduler': self.scheduler
        }

    def train_step(self, images):
        """
        The input tensor represents RGB images
        with pixel values in the [0, 1] range.

        Arguments:
            images: a float tensor with shape [b, 3, h, w].
        Returns:
            a dict with float numbers.
        """

        restored_images, _, _ = self.generator(images)
        content_loss, style_loss = self.perceptual_loss(restored_images[-1], images)
        total_loss = content_loss + 25000.0 * style_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # decay the learning rate
        self.scheduler.step()

        result = {
            'content_loss': content_loss.item(),
            'style_loss': style_loss.item(),
            'total_loss': total_loss.item()
        }

        return result

    def __call__(self, images):
        """
        Inference with trained model.
        """
        self.generator.eval()
        self.generator.requires_grad_(False)

        restored_images = self.generator(images)[0][-1]

        self.generator.train()
        self.generator.requires_grad_(True)

        return restored_images

    def save(self, path):
        """
        Write the current full training state to a folder.
        """
        def save_state(x, name):
            p = os.path.join(path, name + '.pth')
            torch.save(x.state_dict(), p)

        for n, x in self.states.items():
            save_state(x, n)

    def load(self, path):
        """
        Restore full training state from a checkpoint.
        """
        def load_state(x, name):
            p = os.path.join(path, name + '.pth')
            state = torch.load(p, map_location=self.device)
            x.load_state_dict(state)

        for n, x in self.states.items():
            load_state(x, n)
