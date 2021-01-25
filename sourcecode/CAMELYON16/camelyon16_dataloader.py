from sourcecode.dataloader_utils import *

from PIL import ImageColor
from torch.utils.data import Dataset

import boto3
import os.path

import pandas as pd
import xml.etree.ElementTree as et

from sourcecode.wsi_image_utils import *
from sourcecode.logger_utils import *


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
        should_augment = (self.augmentation and fname in self.used_images)
        self.used_images.add(fname)

        x, y = data_augmentation(image, mask, self.img_input_size, self.img_output_size, should_augment)
        #x, y = data_augmentation(image, mask, self.img_input_size, self.img_output_size, self.augmentation)
        return x, y, fname, image.size


def load_dataset(img_dir, img_input_size, dataset_type):

    images = []
    classes = ["normal", "tumor", "test"]
    dataset_root_dir = "{}/{}".format(img_dir, dataset_type)
    logger.info("[{}] {}".format(dataset_type, dataset_root_dir))

    for root, d, _ in sorted(os.walk(dataset_root_dir)):

        for cls in sorted([cls for cls in d if cls in classes]):

            class_root_dir = "{}/{}/patch/{}x{}".format(dataset_root_dir, cls, img_input_size[0], img_input_size[1])
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

                            if is_valid_file(path_img):
                                item = (path_img, path_mask, fname)
                                images.append(item)

    return images


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

