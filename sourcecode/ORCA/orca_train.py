import datetime
import os
import sys
import time
import copy
import csv

import torch.optim as optim
from torch.autograd import Variable

current_path = os.path.abspath('.')
root_path = os.path.dirname(os.path.dirname(current_path))
sys.path.append(root_path)

from sourcecode.ORCA.orca_dataloader import *
from sourcecode.unet_model import *


def train_model(dataloaders,
                model=None,
                patch_size=(640, 640),
                n_epochs=1,
                batch_size=1,
                use_cuda=True,
                output_dir="../../models"):

    # Checking for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if use_cuda else "cpu"
    logger.info('Runing on: {} | available? {}'.format(device, torch.cuda.is_available()))

    torch.cuda.empty_cache()
    if model is None:
        model = UNet(in_channels=3, out_channels=1, padding=True, img_input_size=patch_size).to(device)

    model.train()
    criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(model.parameters())
    optimizer.zero_grad()

    since = time.time()
    qtd_images = 0
    start_epoch = 1
    dataset_train_size = len(dataloaders['train'].dataset)
    for epoch in range(start_epoch, n_epochs + 1):

        time_elapsed = time.time() - since

        logger.info("")
        logger.info('Epoch {}/{} ({:.0f}m {:.0f}s) {}'.format(epoch, n_epochs, time_elapsed // 60, time_elapsed % 60,
                                                              datetime.datetime.now()))
        logger.info("-" * 20)

        for batch_idx, (data, target, fname, original_size) in enumerate(dataloaders['train']):

            logger.info("\tfname: '{}' {}".format(fname[0], (batch_idx + 1)))

            data = Variable(data.to(device))
            target = Variable(target.to(device)).unsqueeze(1)
            # target = Variable(target.to(device))
            # print('X     --> {}'.format(data.size()))
            # print('y     --> {}'.format(target.size()))
            # print('          {}'.format(target))

            optimizer.zero_grad()
            output = model(data)
            # output = model(data).squeeze(0)
            # print('y_hat --> {}'.format(output.size()))
            # print('          {}'.format(output))

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            # break

            qtd_images = (batch_idx + 1) * len(data)
            if batch_idx == 0 or ((batch_idx + 1) % (dataset_train_size / 20) == 0):
                logger.info('\tBatch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    (batch_idx + 1),
                    qtd_images,
                    dataset_train_size,
                    100. * (((batch_idx + 1) * len(data)) / dataset_train_size),
                    loss.item()))

            if qtd_images % 500 == 0:
                save_model(output_dir, model, patch_size, epoch, qtd_images , batch_size, optimizer, loss)

            if loss.item() < 0.0000001 or math.isnan(loss.item()):
                logger.warn("\tBatch: {} (too little loss: {:.15f})".format((batch_idx + 1), loss.item()))
                break

        # save the model - each epoch
        save_model(output_dir, model, patch_size, epoch, qtd_images , batch_size, optimizer, loss)

        # print("\tLoss: {:.6f}".format(loss.item()))
        if loss.item() < 0.0000001 or math.isnan(loss.item()):
            logger.warn("\tEpoch: {} (too little loss: {:.15f})".format(epoch, loss.item()))
            break

    time_elapsed = time.time() - since
    logger.info('-' * 20)
    logger.info('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    save_model(output_dir, model, patch_size, epoch, qtd_images, batch_size, optimizer, loss)


def train_model_with_validation(dataloaders,
                                model=None,
                                patch_size=(640, 640),
                                n_epochs=1,
                                batch_size=1,
                                use_cuda=True,
                                output_dir="../../models"):

    # Checking for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if use_cuda else "cpu"
    logger.info('Runing on: {} | GPU available? {}'.format(device, torch.cuda.is_available()))

    torch.cuda.empty_cache()
    if model is None:
        model = UNet(in_channels=3, out_channels=1, padding=True, img_input_size=patch_size).to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(model.parameters())
    optimizer.zero_grad()

    since = time.time()
    qtd_images = 0
    start_epoch = 1
    with open("../../datasets/ORCA/training/training_loss.csv", mode='a+') as csv_file:

        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['model', 'phase', 'epoch', 'loss', 'date'])

        for epoch in range(start_epoch, n_epochs + 1):

            time_elapsed = time.time() - since

            logger.info("")
            logger.info("-" * 20)
            logger.info('Epoch {}/{} ({:.0f}m {:.0f}s) {}'.format(epoch, n_epochs, time_elapsed // 60, time_elapsed % 60,
                                                                  datetime.datetime.now()))
            logger.info("-" * 20)

            # Each epoch has a training and validation phase
            epoch_loss = {}
            for phase in ['train', 'valid']:

                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
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
                        # statistics
                        running_loss += loss.item() * data.size(0)
                        qtd_images = (batch_idx + 1) * len(data) if phase == 'train' else qtd_images

                epoch_loss[phase] = running_loss / len(dataloaders[phase].dataset)

            # save the model - each epoch
            filename = save_model(output_dir, model, patch_size, epoch, qtd_images , batch_size, optimizer, loss)

            logger.info("-" * 20)
            for phase in ['train', 'valid']:
                print('[{}] Loss: {:.6f}'.format(phase, epoch_loss[phase]))
                csv_writer.writerow([filename, phase, epoch, epoch_loss[phase], time.time()])

    time_elapsed = time.time() - since
    logger.info('-' * 20)
    logger.info('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    save_model(output_dir, model, patch_size, epoch, qtd_images, batch_size, optimizer, loss)


def save_model(model_dir, model, patch_size, epoch, imgs, batch_size, optimizer, loss):
    """
    Save the trained model
    """
    #filename = 'ORCA__Size-{}x{}_Epoch-{}_Images-{}_Batch-{}.pth'.format(patch_size[0], patch_size[1], epoch, imgs, batch_size)
    filename = 'OCDC+ORCA__Size-{}x{}_Epoch-{}_Images-{}_Batch-{}.pth'.format(patch_size[0], patch_size[1], epoch, imgs, batch_size)
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

    dataset_dir = "/media/dalifreire/CCB60537B6052394/Users/Dali/Downloads/ORCA" #"../../datasets/ORCA"
    model_dir = "../../models"

    batch_size = 1
    patch_size = (640, 640)
    color_model = "LAB"
    dataloaders = create_dataloader(tile_size="{}x{}".format(patch_size[0], patch_size[1]),
                                    batch_size=batch_size,
                                    shuffle=True,
                                    img_input_size=patch_size,
                                    img_output_size=patch_size,
                                    dataset_dir=dataset_dir,
                                    color_model=color_model)
    dataset_train_size = len(dataloaders['train'].dataset)
    dataset_test_size = len(dataloaders['test'].dataset)

    # loads our u-net based model to continue previous training
    trained_model_version = "Epoch-1_Images-4181_Batch-1"
    trained_model_path="{}/{}".format(model_dir, 'ORCA__Size-{}x{}_{}.pth'.format(patch_size[0], patch_size[1], trained_model_version))
    #trained_model_path="{}/{}".format(model_dir, 'OCDC__Size-640x640_Color-LAB_Epoch-500_Images-840_Batch-1.pth')
    model = load_checkpoint(file_path=trained_model_path, img_input_size=patch_size, use_cuda=True)

    # starts the training from scratch
    #model = None

    # train the model
    train_model(dataloaders=dataloaders, model=model, n_epochs=100)
