import os
import sys

current_path = os.path.abspath('.')
root_path = os.path.dirname(os.path.dirname(current_path))
sys.path.append(root_path)

from sourcecode.dataloader_utils import *

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

import os.path


class ORCADataset(Dataset):

    def __init__(self,
                 img_dir="../../datasets/OCDC",
                 img_input_size=(640, 640),
                 img_output_size=(640, 640),
                 dataset_type="training",
                 augmentation=None,
                 augmentation_strategy="random",
                 color_model="LAB",
                 start_epoch=1):
        self.img_dir = img_dir
        self.img_input_size = img_input_size
        self.img_output_size = img_output_size
        self.dataset_type = dataset_type
        self.augmentation = augmentation
        self.augmentation_strategy = augmentation_strategy
        self.color_model = color_model
        self.samples = load_dataset(img_dir, img_input_size, dataset_type)
        self.used_images = set()
        self.epoch = start_epoch

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path_img, path_mask, fname = self.samples[idx]
        image = load_pil_image(path_img, False, self.color_model)
        mask = load_pil_image(path_mask) if is_valid_file(path_mask) else None

        x, y, fname, original_size = self.transform(image, mask, fname)
        return [x, y if mask is not None else path_mask, fname, original_size]

    def transform(self, image, mask, fname):

        target_img = None
        GAN_model = None

        if fname in self.used_images:
            self.epoch += 1
            self.used_images.clear()
        self.used_images.add(fname)

        augmentation_operations = []
        
        if self.augmentation_strategy == "no_augmentation" or self.epoch == 1:

            augmentation_operations = None
        
        elif 'color_augmentation' in self.augmentation_strategy:
            
            augmentation_operations.append("color_transfer")

        elif 'inpainting_augmentation' in self.augmentation_strategy:

            augmentation_operations.append("inpainting")

        elif 'random' in self.augmentation_strategy:

            augmentation_operations = self.augmentation.copy()
            augmentation_operations.remove(None)

        elif 'one_by_epoch' in self.augmentation_strategy:

            idx = (self.epoch-1) % len(self.augmentation)
            augmentation_operations.append(self.augmentation[idx])

        if self.epoch > 1 and 'color_transfer' in augmentation_operations:

            target_img_idx = random.randrange(len(self.samples))
            path_img_target, path_mask_target, fname_target = self.samples[target_img_idx-1]
            target_img = load_pil_image(path_img_target, False, self.color_model)

        if self.epoch > 1 and 'inpainting' in augmentation_operations:

            # Prepares the GAN model            
            sourcecode_dir = os.path.dirname(os.path.abspath('.'))
            config_file = os.path.join(sourcecode_dir, 'GAN/configs/config_imagenet_ocdc.yaml')
            config = get_config(config_file)
            checkpoint_path = os.path.join(sourcecode_dir, 'GAN/checkpoints', config['dataset_name'], config['mask_type'] + '_' + config['expname'])

            cuda = config['cuda'] and torch.cuda.is_available()
            device_ids = config['gpu_ids']
            GAN_model = Generator(config['netG'], cuda, device_ids)
            last_model_name = get_model_list(checkpoint_path, "gen", iteration=436800)
            
            checkpoint = torch.load(last_model_name) if torch.cuda.is_available() else torch.load(last_model_name, map_location=lambda storage, loc: storage)
            GAN_model.load_state_dict(checkpoint)

        if len(self.used_images) <= 1:
            logger.info("Epoch: '{}' augmentation {} {}".format(self.epoch, self.augmentation_strategy,
                                                                augmentation_operations))
        
        #x, y = data_augmentation(image, mask, self.img_input_size, self.img_output_size, should_augment)
        #x, y = data_augmentation(image, mask, self.img_input_size, self.img_output_size, False)
        x, y, used_augmentations = data_augmentation(image, target_img, mask, self.img_input_size, self.img_output_size, augmentation_operations, GAN_model)
        return x, y, fname, image.size


def load_dataset(img_dir, img_input_size, dataset_type):

    #first_train = ""
    #with open("application.log", 'r') as file:
    #    first_train = file.read()

    images = []
    classes = ["tumor"]
    dataset_root_dir = "{}/{}".format(img_dir, dataset_type)
    logger.info("[{}] {}".format(dataset_type, dataset_root_dir))

    for root, d, _ in sorted(os.walk(dataset_root_dir)):

        for cls in sorted([cls for cls in d if cls in classes]):

            class_root_dir = "{}/{}/patch/{}x{}".format(dataset_root_dir, cls, img_input_size[0], img_input_size[1])

            for _, img_dir, _ in sorted(os.walk(class_root_dir)):

                for img_number in sorted(img_n for img_n in img_dir):

                    for patch_type in ["01-roi", "02-non_roi"]:

                        original_dir = "{}/{}/{}/01-original".format(class_root_dir, img_number, patch_type)
                        mask_dir = "{}/{}/{}/02-mask".format(class_root_dir, img_number, patch_type)
                        for _, _, fnames in sorted(os.walk(original_dir)):
                            for fname in sorted(fnames):

                                path_img = os.path.join(original_dir, fname)
                                path_mask = os.path.join(mask_dir, fname)

                                #if is_valid_file(path_img) and first_train.find(fname) < 0:
                                if is_valid_file(path_img):
                                    item = (path_img, path_mask, fname)
                                    images.append(item)
    #first_train = ""
    return images


def create_dataloader(tile_size="640x640",
                      batch_size=1,
                      shuffle=False,
                      img_input_size=(640, 640),
                      img_output_size=(640, 640),
                      dataset_dir="../../datasets/ORCA",
                      color_model="LAB",
                      augmentation=None,
                      augmentation_strategy="random",
                      start_epoch=1,
                      validation_split=0.0):

    if augmentation is None:
        augmentation = [None, "horizontal_flip", "vertical_flip", "rotation", "transpose", "elastic_transformation",
                        "grid_distortion", "optical_distortion", "color_transfer", "inpainting"]

    image_datasets = {x: ORCADataset(img_dir=dataset_dir,
                                     img_input_size=img_input_size, img_output_size=img_output_size,
                                     dataset_type='testing' if x == 'test' else 'training',
                                     augmentation=augmentation,
                                     augmentation_strategy='no_augmentation' if x != 'train' else augmentation_strategy,
                                     color_model=color_model,
                                     start_epoch=start_epoch) for x in ['train', 'valid', 'test']}
    if validation_split > 0:

        train_dataset_index, valid_dataset_index = train_test_split(range(len(image_datasets['train'])),
                                                                    test_size=validation_split, random_state=0)
        train_dataset_index.sort()
        valid_dataset_index.sort()

        train_dataset_samples = [image_datasets['train'].samples[index] for index in train_dataset_index]
        valid_dataset_samples = [image_datasets['train'].samples[index] for index in valid_dataset_index]

        image_datasets['train'].samples = train_dataset_samples
        image_datasets['valid'].samples = valid_dataset_samples

    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=shuffle, num_workers=0) for x
        in ['train', 'valid', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}

    if validation_split <= 0:
        del image_datasets['valid']
        del dataloaders['valid']
        del dataset_sizes['valid']

    logger.info("Train images ({}): {} augmentation: {}".format(tile_size, dataset_sizes['train'], augmentation_strategy))
    if validation_split > 0:
        logger.info("Valid images ({}): {} augmentation: {}".format(tile_size, dataset_sizes['valid'], 'no_augmentation'))
    logger.info("Test images ({}): {} augmentation: {}".format(tile_size, dataset_sizes['test'], 'no_augmentation'))

    return dataloaders
