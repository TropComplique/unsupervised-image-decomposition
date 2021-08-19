import os
import time
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from dataset import Images
from model import Model


SIZE = 128
BATCH_SIZE = 32
NUM_EPOCHS = 10
DATA = '/home/dan/datasets/ffhq_256/'
DEVICE = torch.device('cuda:0')
WEIGHTS = 'weights/'

SAVE_STEP = 10000
IMAGE_STEP = 500
PRINT_FREQ = 10


def main():

    dataset = Images(folder=DATA, size=SIZE)
    data_loader = DataLoader(dataset, BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    num_steps = NUM_EPOCHS * (len(dataset) // BATCH_SIZE)
    model = Model(DEVICE, num_steps)

    writer = SummaryWriter(WEIGHTS)
    random_images = []  # images to show in tensorboard
    indices = np.random.randint(0, len(dataset), size=10)

    random_images = [dataset[i] for i in indices]
    random_images = torch.stack(random_images).to(DEVICE)
    random_images = random_images.float().div_(255.0)

    # number of weight updates
    step = 1

    for epoch in range(NUM_EPOCHS):
        for images in data_loader:

            images = images.to(DEVICE)
            images = images.float().div_(255.0)
            # it has shape [b, 3, h, w]

            start = time.perf_counter()
            losses = model.train_step(images)
            step_time = time.perf_counter() - start
            step_time = round(1000 * step_time, 1)

            if step % IMAGE_STEP == 0:

                images = torch.cat([
                    random_images.cpu(),
                    model(random_images).cpu()
                ])

                grid = make_grid(images, nrow=10, padding=0)
                writer.add_image('samples', grid, step)

            if step % PRINT_FREQ == 0:

                for k, v in losses.items():
                    writer.add_scalar(f'losses/{k}', v, i)

                writer.add_scalar('time', step_time, step)
                writer.add_scalar('lr', model.optimizer.param_groups[0]['lr'], step)

            if step % SAVE_STEP == 0:

                save_folder = os.path.join(WEIGHTS, 'latest')
                os.makedirs(save_folder, exist_ok=True)
                model.save(save_folder)

                print(f'model is saved, step {step}')

            step += 1


main()
