from pytorch_caney.data.transforms import SimmimTransform
from pytorch_caney.models.mim.mim import build_mim_model

from utils import create_logger
from config import get_config

import torch
from torch.profiler import profile, ProfilerActivity
from torch.utils.data import Dataset

from timm.utils import AverageMeter

import numpy as np
import time
import datetime
import sys
import argparse


class RandomDataGenerator(Dataset):
    def __init__(self, num_samples, image_size, num_channels, transform):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_channels = num_channels
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        random_image = np.random.randn(self.image_size, self.image_size, self.num_channels)
        random_image = random_image.astype(np.float32)
        img, mask = self.transform(random_image)
        return img, mask 


def parse_args():
    """
    Parse command-line arguments
    """
    parser = argparse.ArgumentParser(
        'pytorch-caney implementation of MiM pre-training script',
        add_help=False)

    parser.add_argument(
        '--cfg',
        type=str,
        required=True,
        metavar="FILE",
        help='path to config file')

    parser.add_argument(
        '--batch-size',
        type=int,
        help="batch size for single GPU")

    parser.add_argument(
        '--num_samples',
        type=int,
        help="Size of dataset",
        default=2_000_000,
    )

    parser.add_argument(
        '--use-checkpoint',
        action='store_true',
        help="whether to use gradient checkpointing to save memory")

    parser.add_argument(
        '--dtype',
        type=str,
        default='float32',
        help='dtype, must be float32, float16, bfloat16',
    )

    args = parser.parse_args()

    config = get_config(args)

    return args, config


def train(config,
          dataloader,
          model,
          optimizer,
          lr_scheduler,
          target_dtype):
    """
    Start pre-training a specific model and dataset.

    Args:
        config: config object
        dataloader: dataloader to use
        model: model to pre-train
        model_wo_ddp: model to pre-train that is not the DDP version
        optimizer: pytorch optimizer
        lr_scheduler: learning-rate scheduler
    """

    logger.info("Start training")

    logger.info(f'Target dtype: {target_dtype}')

    torch.cuda.empty_cache()

    if target_dtype == torch.float32:
        target_dtype = None

    throughput_meter = AverageMeter()

    start_time = time.time()

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):

        start = time.time()

        execute_one_epoch(config, model, dataloader,
                          optimizer, epoch, target_dtype, lr_scheduler,
                          throughput_meter)

        epoch_time = time.time() - start
        logger.info(
            f"EPOCH {epoch} training takes " +
            f"{datetime.timedelta(seconds=int(epoch_time))}")

    total_time = time.time() - start_time

    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    logger.info('Training time {}'.format(total_time_str))
    logger.info(f'Training average throughput {throughput_meter.avg:.2f} samples/second')


def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print(output)

def execute_one_epoch(config,
                      model,
                      dataloader,
                      optimizer,
                      epoch,
                      target_dtype,
                      lr_scheduler,
                      throughput_meter):
    """
    Execute training iterations on a single epoch.

    Args:
        config: config object
        model: model to pre-train
        dataloader: dataloader to use
        optimizer: pytorch optimizer
        epoch: int epoch number
        target_dtype: torch dtype, should match model dtype
        device: device to move inputs to
    """
    model.train()

    optimizer.zero_grad()

    ntrain = config.DATA.NUM_SAMPLES  
    num_steps = max(1,
                    ntrain // (config.DATA.BATCH_SIZE))

    # Set up logging meters
    batch_time = AverageMeter()
    batch_size = config.DATA.BATCH_SIZE

    start = time.time()
    end = time.time()
    with profile(
        activities=[ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            skip_first=20,
            wait=5,
            warmup=10,
            active=5,
            repeat=2),
        on_trace_ready=trace_handler
    ) as profiler:
        for idx, (img, mask) in enumerate(dataloader):

            tp_start = time.time()
            img = img.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)

            if target_dtype:
                img = img.to(target_dtype)

            loss = model(img, mask)

            loss.backward()

            optimizer.step()

            lr_scheduler.step()

            profiler.step()

            throughput = (batch_size / (time.time() - tp_start))
            throughput_meter.update(throughput)
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % config.PRINT_FREQ == 0:
                lr = optimizer.param_groups[0]['lr']
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                etas = batch_time.avg * (num_steps - idx)
                logger.info(
                    f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                    f' eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                    f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                    f'throughput {throughput_meter.val:.4f} samp/s ({throughput_meter.avg:.4f} samp/s)\t'
                    f'mem {memory_used:.0f}MB')


def main(config):
    """
    Starts training process after building the proper model, optimizer, etc.

    Args:
        config: config object
    """

    logger.info('In main')

    target_dtype_str = config.DATA.DTYPE
    if target_dtype_str == 'float32':
        target_dtype = torch.float32
    elif target_dtype_str == 'float16':
        target_dtype = torch.float16
    elif target_dtype_str == 'bfloat16':
        target_dtype = torch.bfloat16

    transform = SimmimTransform(config)

    dataset = RandomDataGenerator(num_samples=config.DATA.NUM_SAMPLES,
                                  image_size=config.DATA.IMG_SIZE,
                                  transform=transform,
                                  num_channels=config.MODEL.SWINV2.IN_CHANS,)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=10,
        shuffle=False,
        pin_memory=True,)

    logger.info(f'MODEL CHECKPOINTING: {config.TRAIN.USE_CHECKPOINT}')

    simmim_model = build_model(config, target_dtype, logger)

    # Count the total number of parameters
    total_params = sum(p.numel() for p in simmim_model.parameters())
    logger.info(f"Total number of parameters: {total_params}")

    # Count the total number of trainable parameters
    trainable_params = sum(p.numel() for p in simmim_model.parameters()
                           if p.requires_grad)
    logger.info(f"Total number of trainable parameters: {trainable_params}")

    ntrain = config.DATA.NUM_SAMPLES 
    num_steps = max(
        1,
        ntrain // (config.DATA.BATCH_SIZE))
    total_steps = num_steps * config.TRAIN.EPOCHS
    logger.info(f'Number of steps: {num_steps}')

    optimizer = torch.optim.AdamW(simmim_model.parameters(), lr=config.TRAIN.BASE_LR)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=config.TRAIN.BASE_LR,
                                                    total_steps=total_steps,
                                                    pct_start=0.4)
    logger.info('Starting training block')

    train(config,
          dataloader,
          simmim_model,
          optimizer,
          lr_scheduler=scheduler,
          target_dtype=target_dtype)


def build_model(config, dtype, logger):

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")

    model = build_mim_model(config)

    model = model.cuda()

    if dtype != torch.float:
        model = model.to(dtype)
    logger.info(f'Cast model to {dtype}')

    return model


def setup_seeding(config):
    seed = config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)


if __name__ == '__main__':

    _, config = parse_args()

    setup_seeding(config)

    logger = create_logger(dist_rank=0,
                           name=f"{config.MODEL.NAME}")

    torch.backends.cudnn.benchmark = True

    sys.exit(main(config))

