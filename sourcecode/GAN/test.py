import os
import random
import time
import shutil
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import dataset
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
#from torch.utils.tensorboard import SummaryWriter

from trainer import Trainer
from data.dataset import Dataset
from utils.tools import get_config, random_bbox, mask_image, get_model_list
from utils.logger import get_logger

from model.networks import Generator

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml', help="training configuration")
parser.add_argument('--seed', type=int, help='manual seed')
parser.add_argument('--checkpoint_path', type=str, default='')
parser.add_argument('--iter', type=int, default=0)


def main():
    
    torch.autograd.set_detect_anomaly(True)
    torch.cuda.empty_cache()

    args = parser.parse_args()
    config = get_config(args.config)

    # CUDA configuration
    cuda = config['cuda']
    device_ids = config['gpu_ids']
    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
        device_ids = list(range(len(device_ids)))
        config['gpu_ids'] = device_ids
        cudnn.benchmark = True

    # Configure checkpoint path
    if not args.checkpoint_path:
        checkpoint_path = os.path.join('checkpoints',
                                        config['dataset_name'],
                                        config['mask_type'] + '_' + config['expname'])
    else:
        checkpoint_path = args.checkpoint_path

    logger = get_logger(checkpoint_path)

    last_model_name = get_model_list(checkpoint_path, "gen", iteration=args.iter)
    logger.info("Using model: '{}'".format(last_model_name))

    results_dir = os.path.join(checkpoint_path, "results__{}".format(last_model_name.split("/")[-1]))[:-3]
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Set random seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    logger.info("Random seed: {}".format(args.seed))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed_all(args.seed)

    # Log the arguments
    logger.info("Arguments: {}".format(args))

    # Log the configuration
    logger.info("Configuration: {}".format(config))

    try:  # for unexpected error logging
    
        # Load the dataset
        test_dataset = Dataset(data_path=config['test_data_path'],
                               with_subfolder=config['data_with_subfolder'],
                               image_shape=config['image_shape'],
                               random_crop=config['random_crop'],
                               return_name=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=config['batch_size'],
                                                  shuffle=False,
                                                  num_workers=config['num_workers'])


        dataset_size = len(test_dataset)
        logger.info("Testing on dataset: {} ({} images)".format(config['dataset_name'], dataset_size))

        iterable_test_loader = iter(test_loader)
        for iteration in range(1, dataset_size + 1):
        
            try:
                fname, ground_truth = next(iterable_test_loader)
            except StopIteration:
                iterable_test_loader = iter(test_loader)
                fname, ground_truth = next(iterable_test_loader)

            logger.info("Iteration {}: {}".format(iteration, fname))

            bboxes = random_bbox(config, batch_size=ground_truth.size(0))
            x, mask = mask_image(ground_truth, bboxes, config)
                        
            # Define the trainer
            netG = Generator(config['netG'], cuda, device_ids)
            # Resume weight
            checkpoint = torch.load(last_model_name) if torch.cuda.is_available() else torch.load(last_model_name, map_location=lambda storage, loc: storage)
            netG.load_state_dict(checkpoint)
            model_iteration = int(last_model_name[-11:-3])
            
            if cuda:
                netG = nn.parallel.DataParallel(netG, device_ids=device_ids)
                x = x.cuda()
                mask = mask.cuda()
                ground_truth = ground_truth.cuda()

            # Inference
            x1, x2, offset_flow = netG(x, mask)
            inpainted_result = x2 * mask + x * (1. - mask)

            viz_images = torch.stack([x, inpainted_result, offset_flow], dim=1)
            viz_images = viz_images.view(-1, *list(x.size())[1:])
            vutils.save_image(viz_images,
                                '%s/result_%03d.png' % (results_dir, iteration),
                                nrow=3 * 4,
                                normalize=True)

            torch.cuda.empty_cache()

    except Exception as e:  # for unexpected error logging
        logger.error("{}".format(e))
        raise e


if __name__ == '__main__':
    main()
