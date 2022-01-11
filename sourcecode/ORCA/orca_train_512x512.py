import datetime
import os
import sys
import time
import csv

import torch.optim as optim
from torch.autograd import Variable

current_path = os.path.abspath('.')
root_path = os.path.dirname(os.path.dirname(current_path))
sys.path.append(root_path)

from sourcecode.ORCA.orca_dataloader_512x512 import *
from sourcecode.unet_model import *


def train_model_with_validation(dataloaders,
                                model=None,
                                patch_size=(512, 512),
                                n_epochs=1,
                                batch_size=1,
                                use_cuda=True,
                                output_dir="../../models",
                                augmentation_strategy="random",
                                augmentation_operations=[None],
                                result_file_csv="../../datasets/ORCA_512x512/training/orca_training_accuracy_loss.csv"):

    # Checking for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if use_cuda else "cpu"
    logger.info('Runing on: {} | GPU available? {}'.format(device, torch.cuda.is_available()))

    torch.cuda.empty_cache()
    if model is None:
        model = UNet(in_channels=3, out_channels=1, padding=True, img_input_size=patch_size).to(device)

    augmentation = augmentation_strategy if augmentation_strategy in ["no_augmentation", "color_augmentation", "inpainting_augmentation"] else "{}_{}_operations".format(augmentation_strategy, len(augmentation_operations)-1)
    with open(result_file_csv, mode='a+') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['model', 'augmentation', 'phase', 'epoch', 'loss', 'accuracy', 'date', 'transformations'])

    criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(model.parameters())
    optimizer.zero_grad()

    best_loss = 1.0
    best_acc = 0.0

    since = time.time()
    qtd_images = 0
    start_epoch = 367
    for epoch in range(start_epoch, n_epochs + 1):

        time_elapsed = time.time() - since

        logger.info("")
        logger.info("-" * 20)
        logger.info('Epoch {}/{} {} ({:.0f}m {:.0f}s) {}'.format(epoch, n_epochs, augmentation, time_elapsed // 60,
                                                                 time_elapsed % 60,
                                                                 datetime.datetime.now()))
        logger.info("-" * 20)

        # Each epoch has a training and validation phase
        epoch_loss = {}
        epoch_acc = {}
        for phase in ['train', 'test']:

            model.train()
            #if phase == 'train':
            #    model.train()  # Set model to training mode
            #else:
            #    model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_accuracy = 0
            for batch_idx, (data, target, fname, original_size) in enumerate(dataloaders[phase]):

                logger.info("\tfname: '{}' {}".format(fname[0], (batch_idx + 1)))

                data = Variable(data.to(device))
                target = Variable(target.to(device)).unsqueeze(1)
                # target = Variable(target.to(device))
                # print('X     --> {}'.format(data.size()))
                # print('y     --> {}'.format(target.size()))
                # print('          {}'.format(target))

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):

                    output = model(data)
                    # output = model(data).squeeze(0)
                    # print('y_hat --> {}'.format(output.size()))
                    # print('          {}'.format(output))

                    loss = criterion(output, target)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    torch.cuda.empty_cache()

                    preds = torch.zeros(output.size(), dtype=torch.double).cuda()
                    preds[output >= 0.5] = 1.0

                    # statistics
                    acc = torch.sum(preds == target.data).detach().cpu().numpy() / (data.size(0)*data.size(-1)*data.size(-2))
                    running_loss += loss.item() * data.size(0)
                    running_accuracy += acc

                    qtd_images = (batch_idx + 1) * len(data) if phase == 'train' else qtd_images

            epoch_loss[phase] = running_loss / len(dataloaders[phase].dataset)
            epoch_acc[phase] = running_accuracy / len(dataloaders[phase].dataset)

        # save the model - each epoch
        #if epoch == 1 or epoch_loss[phase] < best_loss or epoch_acc[phase] > best_acc or (epoch % 10 == 0):
        filename = save_model(output_dir, model, patch_size, epoch, qtd_images , batch_size, augmentation, optimizer, loss)

        if epoch_loss[phase] < best_loss:
            best_loss = epoch_loss[phase]
        if epoch_acc[phase] > best_acc:
            best_acc = epoch_acc[phase]

        logger.info("-" * 20)

        with open(result_file_csv, mode='a+') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for phase in ['train', 'test']:
                print('[{}] Loss: {:.6f}'.format(phase, epoch_loss[phase]))
                csv_writer.writerow([filename, augmentation, phase, epoch, epoch_loss[phase], epoch_acc[phase], datetime.datetime.now(), str(augmentation_operations).replace(",", "")])

    time_elapsed = time.time() - since
    logger.info('-' * 20)
    logger.info('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.info('Best accuracy: {}'.format(best_acc))

    save_model(output_dir, model, patch_size, epoch, qtd_images, batch_size, augmentation_strategy, optimizer, loss)


def save_model(model_dir, model, patch_size, epoch, imgs, batch_size, augmentation_strategy, optimizer, loss):
    """
    Save the trained model
    """
    filename = 'ORCA_512x512__Size-{}x{}_Epoch-{}_Images-{}_Batch-{}__{}_geo.pth'.format(patch_size[0], patch_size[1], epoch, imgs, batch_size, augmentation_strategy)
    logger.info("Saving the model: '{}'".format(filename))

    filepath = os.path.join(model_dir, filename) if model_dir is not None else filename
    with open(filepath, 'wb') as f:
        torch.save({
            'epoch': epoch,
            'batch_size': batch_size,
            'dataset': 'ORCADataset',
            'model_in_channels': model.model_input_channels(),
            'model_out_channels': model.model_output_channels(),
            'model_up_mode': model.model_up_mode(),
            'model_padding': model.model_padding(),
            'criterion': 'nn.BCELoss',
            'optimizer': 'optim.Adam',
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, f)
    return filename


if __name__ == '__main__':

    dataset_dir = "../../datasets/ORCA_512x512"
    model_dir = "../../models"
    
    augmentation_strategy = "random" # "no_augmentation", "color_augmentation", "inpainting_augmentation", "standard", "random"
    augmentation = [None, "horizontal_flip", "vertical_flip", "rotation", "transpose"]
    #[None, "horizontal_flip", "vertical_flip", "rotation", "transpose", "elastic_transformation", "grid_distortion", "optical_distortion", "color_transfer", "inpainting"]

    batch_size = 1
    patch_size = (512, 512)
    color_model = "LAB"
    dataloaders = create_dataloader(tile_size="{}x{}".format(patch_size[0], patch_size[1]),
                                    batch_size=batch_size,
                                    shuffle=False,
                                    img_input_size=patch_size,
                                    img_output_size=patch_size,
                                    dataset_dir=dataset_dir,
                                    color_model=color_model,
                                    augmentation=augmentation,
                                    augmentation_strategy=augmentation_strategy,
                                    start_epoch=367,
                                    validation_split=0.0)

    # loads our u-net based model to continue previous training
    trained_model_version = "ORCA_512x512__Size-512x512_Epoch-366_Images-100_Batch-1__random"
    trained_model_path = "{}/{}.pth".format(model_dir, trained_model_version)
    model = load_checkpoint(file_path=trained_model_path, img_input_size=patch_size, use_cuda=True)

    # starts the training from scratch
    # model = None

    # train the model
    result_file_csv = "../../datasets/ORCA_512x512/training/orca_training_accuracy_loss_geo.csv"
    train_model_with_validation(dataloaders=dataloaders,
                                model=model,
                                n_epochs=400,
                                augmentation_strategy=augmentation_strategy,
                                output_dir=model_dir,
                                augmentation_operations=augmentation,
                                result_file_csv=result_file_csv)
