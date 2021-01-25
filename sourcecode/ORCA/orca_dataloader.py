from sourcecode.dataloader_utils import *

from torch.utils.data import Dataset

import os.path


class ORCADataset(Dataset):

    def __init__(self,
                 img_dir="../../datasets/ORCA",
                 img_input_size=(640, 640),
                 img_output_size=(640, 640),
                 dataset_type="training",
                 augmentation=True,
                 color_model="LAB"):
        self.img_dir = img_dir
        self.img_input_size = img_input_size
        self.img_output_size = img_output_size
        self.dataset_type = dataset_type
        self.augmentation = augmentation
        self.color_model = color_model
        self.samples = load_dataset(img_dir, img_input_size, dataset_type)
        self.used_images = set()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path_img, path_mask, fname = self.samples[idx]
        image = load_pil_image(path_img, False, self.color_model)
        mask = load_pil_image(path_mask) if is_valid_file(path_mask) else None

        x, y, fname, original_size = self.transform(image, mask, fname)
        return [x, y if mask is not None else path_mask, fname, original_size]

    def transform(self, image, mask, fname):
        should_augment = (self.augmentation and fname in self.used_images)
        self.used_images.add(fname)

        x, y = data_augmentation(image, mask, self.img_input_size, self.img_output_size, should_augment)
        #x, y = data_augmentation(image, mask, self.img_input_size, self.img_output_size, self.augmentation)
        return x, y, fname, image.size


def load_dataset(img_dir, img_input_size, dataset_type):

    images = []
    classes = ["tumor"]
    dataset_root_dir = "{}/{}".format(img_dir, dataset_type)
    logger.info("[{}] {}".format(dataset_type, dataset_root_dir))

    for root, d, _ in sorted(os.walk(dataset_root_dir)):

        for cls in sorted([cls for cls in d if cls in classes]):

            class_root_dir = "{}/{}/patch/{}x{}".format(dataset_root_dir, cls, img_input_size[0], img_input_size[1])

            for _, img_dir, _ in sorted(os.walk(class_root_dir)):

                for img_number in sorted(img_n for img_n in img_dir):

                    original_dir = "{}/{}/{}/01-original".format(class_root_dir, img_number, "01-roi")
                    mask_dir = "{}/{}/{}/02-mask".format(class_root_dir, img_number, "01-roi")
                    for _, _, fnames in sorted(os.walk(original_dir)):
                        for fname in sorted(fnames):

                            path_img = os.path.join(original_dir, fname)
                            path_mask = os.path.join(mask_dir, fname)

                            if is_valid_file(path_img):
                                item = (path_img, path_mask, fname)
                                images.append(item)

    return images


def create_dataloader(tile_size="640x640",
                      batch_size=1,
                      shuffle=False,
                      img_input_size=(640, 640),
                      img_output_size=(640, 640),
                      dataset_dir="../../datasets/ORCA",
                      color_model="LAB"):
    image_datasets = {x: ORCADataset(img_dir=dataset_dir,
                                     img_input_size=img_input_size, img_output_size=img_output_size,
                                     dataset_type='training' if x == 'train' else 'testing',
                                     augmentation=True if x == 'train' else False,
                                     color_model=color_model) for x in ['train', 'test']}

    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=shuffle, num_workers=0) for x
        in ['train', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

    logger.info("Train images ({}): {} (augmentation: {})".format(tile_size, dataset_sizes['train'], image_datasets['train'].augmentation))
    logger.info("Test images ({}): {} (augmentation: {})".format(tile_size, dataset_sizes['test'], image_datasets['test'].augmentation))
    return dataloaders
