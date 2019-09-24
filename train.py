import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from input_pipeline import Images
from model import Model


BATCH_SIZE = 8
SIZE = 128
DATA = '/home/dan/datasets/posters/images/'
NUM_EPOCHS = 10
DEVICE = torch.device('cuda:1')
MODEL_SAVE_PREFIX = 'models/run00'
LOGS_DIR = 'summaries/run00/'

SAVE_EPOCH = 1
PLOT_IMAGE_STEP = 10
PLOT_LOSS_STEP = 1


def main():

    dataset = Images(folder=DATA, size=SIZE)
    data_loader = DataLoader(
        dataset=dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=4,
        pin_memory=True, drop_last=True
    )
    num_steps = NUM_EPOCHS * (len(dataset) // BATCH_SIZE)
    model = Model(
        image_size=(SIZE, SIZE), batch_size=BATCH_SIZE,
        device=DEVICE, num_steps=num_steps
    )

    writer = SummaryWriter(LOGS_DIR)
    random_images = []  # images to show in tensorboard
    indices = np.random.randint(0, len(dataset), size=10)

    for k, i in enumerate(indices):
        image = dataset[i]
        writer.add_image(f'sample_{k}', image, 0)
        random_images.append(image.unsqueeze(0).to(DEVICE))

    # number of weight updates
    i = 0

    for e in range(1, NUM_EPOCHS + 1):
        for images in data_loader:

            images = images.to(DEVICE)
            # it has shape [b, 3, h, w]

            i += 1
            start = time.perf_counter()
            losses = model.train_step(images)
            step_time = time.perf_counter() - start
            step_time = round(1000 * step_time, 1)

            if i % PLOT_IMAGE_STEP == 0:

                for j, image in enumerate(random_images):
                    with torch.no_grad():
                        restored_image, _, _ = model.G(image)
                        restored_image = restored_image[0].cpu()
                    writer.add_image(f'sample_{j}', restored_image, i)

            if i % PLOT_LOSS_STEP == 0:

                for k, v in losses.items():
                    writer.add_scalar(f'losses/{k}', v, i)

            print(f'epoch {e}, iteration {i}, time {step_time} ms')

        if e % SAVE_EPOCH == 0:
            model.save_model(MODEL_SAVE_PREFIX + f'_epoch_{e}')


main()
