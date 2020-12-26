import torch
import torchvision.transforms.functional as TF

import boto3
import random
import os.path
import matplotlib.pyplot as plt

import pandas as pd
import xml.etree.ElementTree as et

from sourcecode.wsi_image_utils import *
from sourcecode.logger_utils import *

from PIL import ImageColor
from torch.utils.data import Dataset
from albumentations import (
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion
)


class Camelyon16Dataset(Dataset):

    def __init__(self,
                 img_dir="../../datasets/CAMELYON16",
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
        #should_augment = (self.augmentation and fname in self.used_images)
        #self.used_images.add(fname)
        #x, y = data_augmentation(image, mask, self.img_input_size, self.img_output_size, should_augment)

        x, y = data_augmentation(image, mask, self.img_input_size, self.img_output_size, self.augmentation)
        return x, y, fname, image.size


def is_valid_file(filename, extensions=('.jpg', '.bmp', '.tif', '.png')):
    return str(filename).lower().endswith(extensions)


def load_dataset(img_dir, img_input_size, dataset_type):

    first_train = ""
    with open("application.log", 'r') as file:
        first_train = file.read()

    #result_dir = "/media/dalifreire/HD1/DALI/HumanOralDataset+CAMELYON16/results/Epoch-10_Images-109278_Batch-1/LAB_640x640/01-output"
    #result_imgs = os.listdir(result_dir)

    images = []
    classes = ["normal", "tumor", "test"]
    dataset_root_dir = "{}/{}".format(img_dir, dataset_type)
    logger.info("[{}] {}".format(dataset_type, dataset_root_dir))

    for root, d, _ in sorted(os.walk(dataset_root_dir)):

        for cls in sorted([cls for cls in d if cls in classes]):

            class_root_dir = "{}/{}/tiles/{}x{}".format(dataset_root_dir, cls, img_input_size[0], img_input_size[1])
            for _, img_dir, _ in sorted(os.walk(class_root_dir)):

                for img_number in sorted(img_n for img_n in img_dir if img_n.startswith(tuple(classes))):

                    original_dir = "{}/{}/{}/01-original".format(class_root_dir, img_number,
                                                        "01-roi" if class_root_dir.find("tumor") >= 0 else "02-non_roi")
                    mask_dir = "{}/{}/{}/02-mask".format(class_root_dir, img_number,
                                                        "01-roi" if class_root_dir.find("tumor") >= 0 else "02-non_roi")

                    for _, _, fnames in sorted(os.walk(original_dir)):
                        for fname in sorted(fnames):

                            path_img = os.path.join(original_dir, fname)
                            path_mask = os.path.join(mask_dir, fname)

                            #if is_valid_file(path_img) and fname not in result_imgs:
                            #if is_valid_file(path_img) and first_train.find(fname) < 0:
                            if is_valid_file(path_img):
                                item = (path_img, path_mask, fname)
                                images.append(item)
    first_train = ""
    return images


def data_augmentation(input_image, output_mask, img_input_size=(640, 640), img_output_size=(640, 640), aug=True):
    image = TF.resize(input_image, size=img_output_size)
    mask = TF.resize(output_mask, size=img_output_size) if output_mask is not None and np.any(
        np.unique(pil_to_np(output_mask) > 0)) else None

    if aug:

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask) if mask is not None else None

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask) if mask is not None else None

        # Random rotation
        if random.random() > 0.5 and img_input_size[0] == img_input_size[1]:
            augmented = RandomRotate90(p=1)(image=np.array(image),
                                            mask=np.array(mask) if mask is not None else np.zeros(img_output_size))
            image = Image.fromarray(augmented['image'])
            mask = Image.fromarray(augmented['mask'])

        # Random transpose
        if random.random() > 0.5 and img_input_size[0] == img_input_size[1]:
            augmented = Transpose(p=1)(image=np.array(image),
                                       mask=np.array(mask) if mask is not None else np.zeros(img_output_size))
            image = Image.fromarray(augmented['image'])
            mask = Image.fromarray(augmented['mask'])

        # Random elastic transformation
        if random.random() > 0.5:
            alpha = random.randint(100, 200)
            augmented = ElasticTransform(p=1, alpha=alpha, sigma=alpha * 0.05, alpha_affine=alpha * 0.03)(
                image=np.array(image), mask=np.array(mask) if mask is not None else np.zeros(img_output_size))
            image = Image.fromarray(augmented['image'])
            mask = Image.fromarray(augmented['mask'])

        # Random GridDistortion
        if random.random() > 0.5:
            augmented = GridDistortion(p=1)(image=np.array(image),
                                            mask=np.array(mask) if mask is not None else np.zeros(img_output_size))
            image = Image.fromarray(augmented['image'])
            mask = Image.fromarray(augmented['mask'])

        # Random OpticalDistortion
        if random.random() > 0.5:
            augmented = OpticalDistortion(p=1, distort_limit=1, shift_limit=0.5)(image=np.array(image),
                                                                                 mask=np.array(
                                                                                     mask) if mask is not None else np.zeros(
                                                                                     img_output_size))
            image = Image.fromarray(augmented['image'])
            mask = Image.fromarray(augmented['mask'])

    # Transform to grayscale (1 channel)
    mask = TF.to_grayscale(mask, num_output_channels=1) if mask is not None else None

    # Transform to pytorch tensor and binarize the mask
    image = TF.to_tensor(image).float()

    unique_mask_values = np.unique(pil_to_np(mask))
    mask = torch.zeros(img_output_size) if mask is None or not np.any(unique_mask_values) else (
        torch.ones(img_output_size) if np.any(unique_mask_values) and unique_mask_values.size == 1 else TF.to_tensor(
            np_to_pil(basic_threshold(np_img=pil_to_np(mask)))).squeeze(0).float())

    return image, mask


def create_dataloader(tile_size="640x640",
                      batch_size=1,
                      shuffle=False,
                      img_input_size=(640, 640),
                      img_output_size=(640, 640),
                      dataset_dir="../../datasets/CAMELYON16",
                      color_model="LAB"):
    image_datasets = {x: Camelyon16Dataset(img_dir=dataset_dir,
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


def show_image(img):
    if isinstance(img, np.ndarray) or len(img.shape) == 2:
        plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    else:
        rgb = img.permute(1, 2, 0)
        plt.imshow(rgb)


def dataset_show(dataloader, batch_size=6, show_masks=False, row_limit=10):
    cont_rows = 0
    for batch_idx, (images, masks, fname, output_size) in enumerate(dataloader):

        logger.info('Batch {}: {}/{} images {} masks {}'.format((batch_idx + 1),
                                                                (batch_idx + 1) * len(images),
                                                                len(dataloader.dataset),
                                                                images.shape,
                                                                masks.shape))
        # print('\t{}'.format(masks))

        # show 1 line of 'batch_size' images
        fig = plt.figure(figsize=(20, 20))
        for idx in np.arange(batch_size):
            ax = fig.add_subplot(1, batch_size, idx + 1, xticks=[], yticks=[])
            show_image(images[idx])
            ax.set_title("{}".format(fname[idx]))

        if show_masks:
            # show 1 line of 'batch_size' masks
            fig = plt.figure(figsize=(20, 20))
            for idx in np.arange(batch_size):
                ax = fig.add_subplot(1, batch_size, idx + 1, xticks=[], yticks=[])
                show_image(masks[idx])

        cont_rows = cont_rows + 1
        if cont_rows >= row_limit:
            break


def download_s3_folder(bucket_name, s3_folder, local_dir=None):
    """
    Download the contents of a folder directory
    Args:
        bucket_name: the name of the s3 bucket
        s3_folder: the folder path in the s3 bucket
        local_dir: a relative or absolute directory path to save in the local file system
    """
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder):

        target = obj.key if local_dir is None else os.path.join(local_dir, obj.key)

        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))

        if not target.endswith("/"):
            bucket.download_file(obj.key, target)


def xml_annotations_to_dataframe(xml_file):

    dfcols = ['Name', 'Type', 'Group', 'Color', 'Order', 'X', 'Y']
    df = pd.DataFrame(columns=dfcols)

    xml = et.parse(xml_file)
    xml_root = xml.getroot()
    for xml_annotation in xml_root.iter('Annotation'):

        xml_attr_name = xml_annotation.attrib.get('Name')
        xml_attr_type = xml_annotation.attrib.get('Type')
        xml_attr_group = xml_annotation.attrib.get('PartOfGroup')
        xml_attr_color = xml_annotation.attrib.get('Color')
        # print("{} \t {} \t {}".format(xml_attr_name, xml_attr_group, xml_attr_color))

        for xml_coordinate in xml_annotation.iter('Coordinate'):

            xml_attr_order = xml_coordinate.attrib.get('Order')
            xml_attr_x = float(xml_coordinate.attrib.get('X'))
            xml_attr_y = float(xml_coordinate.attrib.get('Y'))

            df = df.append(pd.Series([xml_attr_name,
                                      xml_attr_type,
                                      xml_attr_group,
                                      xml_attr_color,
                                      xml_attr_order,
                                      xml_attr_x,
                                      xml_attr_y], index=dfcols), ignore_index=True)

    for xml_group in xml_root.iter('Group'):
        xml_attr_group_name = xml_group.attrib.get('Name')
        xml_attr_group_color = xml_group.attrib.get('Color')
        df.loc[df.Group == xml_attr_group_name, 'Color'] = xml_attr_group_color

    return df


def find_annotation_contours(annotation_file):

    df = xml_annotations_to_dataframe(annotation_file)
    df.head()

    contours = []
    cont_tumor_regions = 0
    regions = sorted(set(df['Name']))
    for idx, region_name in enumerate(regions):
        annotation_type = df[df['Name'] == region_name]['Type'].iloc[0]
        group = df[df['Name'] == region_name]['Group'].iloc[0]
        if group == "_2" or group == "Exclusion":
            color = ImageColor.getrgb(df[df['Name'] == region_name]['Color'].iloc[0]) + (150,)
        else:
            color = ImageColor.getrgb(df[df['Name'] == region_name]['Color'].iloc[0]) + (100,)
            cont_tumor_regions += 1
        x_coord = df[df['Name'] == region_name]['X']
        y_coord = df[df['Name'] == region_name]['Y']

        points = list(zip(x_coord, y_coord))
        contours.append((region_name, annotation_type, group, color, points))

    return cont_tumor_regions, contours

