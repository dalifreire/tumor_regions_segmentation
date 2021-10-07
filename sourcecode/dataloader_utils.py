import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

import random
import matplotlib.pyplot as plt

current_path = os.path.abspath('.')
root_path = os.path.dirname(os.path.dirname(current_path))
sys.path.append(root_path)

from sourcecode.wsi_image_utils import *
from sourcecode.logger_utils import *
from sourcecode.GAN.model.networks import Generator
from sourcecode.GAN.utils.tools import get_config, random_bbox, mask_image, is_image_file, default_loader, normalize, get_model_list

from torchvision import transforms

from albumentations import (
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion
)


def is_valid_file(filename, extensions=('.jpg', '.bmp', '.tif', '.png')):
    return str(filename).lower().endswith(extensions)


def data_augmentation(input_image, target_img, output_mask, img_input_size=(640, 640), img_output_size=(640, 640), aug=None, GAN_model=None):

    image = TF.resize(input_image, size=img_output_size)
    target_image = TF.resize(target_img, size=img_output_size) if target_img is not None else None
    mask = TF.resize(output_mask, size=img_output_size) if output_mask is not None and np.any(
        np.unique(pil_to_np(output_mask) > 0)) else None

    used_augmentations = []
    if aug is not None and len(aug) > 0:

        # Random horizontal flipping
        if "horizontal_flip" in aug and (len(aug) < 2 or random.random() > 0.5):
            image = TF.hflip(image)
            mask = TF.hflip(mask) if mask is not None else None
            used_augmentations.append("horizontal_flip")

        # Random vertical flipping
        if "vertical_flip" in aug and (len(aug) < 2 or random.random() > 0.5):
            image = TF.vflip(image)
            mask = TF.vflip(mask) if mask is not None else None
            used_augmentations.append("vertical_flip")

        # Random rotation
        if "rotation" in aug and (len(aug) < 2 or random.random() > 0.5) and img_input_size[0] == img_input_size[1]:
            augmented = RandomRotate90(p=1)(image=np.array(image),
                                            mask=np.array(mask) if mask is not None else np.zeros(img_output_size))
            image = Image.fromarray(augmented['image'])
            mask = Image.fromarray(augmented['mask'])
            used_augmentations.append("rotation")

        # Random transpose
        if "transpose" in aug and (len(aug) < 2 or random.random() > 0.5) and img_input_size[0] == img_input_size[1]:
            augmented = Transpose(p=1)(image=np.array(image),
                                       mask=np.array(mask) if mask is not None else np.zeros(img_output_size))
            image = Image.fromarray(augmented['image'])
            mask = Image.fromarray(augmented['mask'])
            used_augmentations.append("transpose")

        # Random elastic transformation
        if "elastic_transformation" in aug and (len(aug) < 2 or random.random() > 0.5):
            alpha = random.randint(100, 200)
            augmented = ElasticTransform(p=1, alpha=alpha, sigma=alpha * 0.05, alpha_affine=alpha * 0.03)(
                image=np.array(image), mask=np.array(mask) if mask is not None else np.zeros(img_output_size))
            image = Image.fromarray(augmented['image'])
            mask = Image.fromarray(augmented['mask'])
            used_augmentations.append("elastic_transformation")

        # Random GridDistortion
        if "grid_distortion" in aug and (len(aug) < 2 or random.random() > 0.5):
            augmented = GridDistortion(p=1)(image=np.array(image),
                                            mask=np.array(mask) if mask is not None else np.zeros(img_output_size))
            image = Image.fromarray(augmented['image'])
            mask = Image.fromarray(augmented['mask'])
            used_augmentations.append("grid_distortion")

        # Random OpticalDistortion
        if "optical_distortion" in aug and (len(aug) < 2 or random.random() > 0.5):
            augmented = OpticalDistortion(p=1, distort_limit=1, shift_limit=0.5)(image=np.array(image),
                                                                                 mask=np.array(
                                                                                     mask) if mask is not None else np.zeros(
                                                                                     img_output_size))
            image = Image.fromarray(augmented['image'])
            mask = Image.fromarray(augmented['mask'])
            used_augmentations.append("optical_distortion")

        # Color transfer augmentation
        if "color_transfer" in aug and target_image is not None and (len(aug) < 2 or random.random() > 0.5):
            
            original_img_lab = TF.to_tensor(image).permute(1, 2, 0).numpy()
            target_img_lab = TF.to_tensor(target_image).permute(1, 2, 0).numpy()

            _, _, augmented_img = transfer_color(original_img_lab, target_img_lab)
            image = transforms.ToPILImage()(torch.from_numpy(augmented_img).permute(2, 0, 1))
            used_augmentations.append("color_transfer")

        # Inpainting augmentation
        if "inpainting" in aug and (len(aug) < 2 or random.random() > 0.5):
            
            width, height = image.size
            sourcecode_dir = os.path.dirname(os.path.abspath('.'))
            config_file = os.path.join(sourcecode_dir, 'GAN/configs/config_imagenet_ocdc.yaml')
            config = get_config(config_file)

            # Setting the points for cropped image
            crop_size = config['image_shape']
            left = np.random.randint(0, width-crop_size[0])
            top = np.random.randint(0, height-crop_size[1])

            img = image.convert(mode="RGB").crop((left, top, left+crop_size[0], top+crop_size[1]))
            img = transforms.ToTensor()(img)
            img = normalize(img)
            img = img.unsqueeze(dim=0)

            bboxes = random_bbox(config, batch_size=img.size(0))
            inpainting_img, inpainting_mask = mask_image(img, bboxes, config)

            if torch.cuda.is_available():
                GAN_model = nn.parallel.DataParallel(GAN_model)
                inpainting_img = inpainting_img.cuda()
                inpainting_mask = inpainting_mask.cuda()

            # Inpainting inference
            x1, x2, offset_flow = GAN_model(inpainting_img, inpainting_mask)
            inpainted_result = x2 * mask + inpainting_img * (1. - inpainting_mask)
            inpainted_result_lab = rgb_to_lab(inpainted_result.permute(1, 2, 0).numpy())

            augmented_img = pil_to_np(image)
            augmented_img[top:top+crop_size[1], left:left+crop_size[0]] = inpainted_result_lab

            image = transforms.ToPILImage()(torch.from_numpy(augmented_img).permute(2, 0, 1))
            used_augmentations.append("inpainting")

    # Transform to grayscale (1 channel)
    mask = TF.to_grayscale(mask, num_output_channels=1) if mask is not None else None

    # Transform to pytorch tensor and binarize the mask
    image = TF.to_tensor(image).float()

    unique_mask_values = np.unique(pil_to_np(mask))
    mask = torch.zeros(img_output_size) if mask is None or not np.any(unique_mask_values) else (
        torch.ones(img_output_size) if np.any(unique_mask_values) and unique_mask_values.size == 1 else TF.to_tensor(
            np_to_pil(basic_threshold(np_img=pil_to_np(mask)))).squeeze(0).float())

    return image, mask, used_augmentations


def show_image(img):
    if isinstance(img, np.ndarray) or len(img.shape) == 2:
        plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    else:
        rgb = img.permute(1, 2, 0)
        plt.imshow(rgb)


def dataset_show(dataloader, batch_size=6, show_masks=False, row_limit=10):
    cont_rows = 0
    for batch_idx, (images, masks, fname, output_size) in enumerate(dataloader):

        # print('\t{}'.format(masks))
        logger.info('Batch {}: {}/{} images {} masks {}'.format((batch_idx + 1),
                                                                (batch_idx + 1) * len(images),
                                                                len(dataloader.dataset),
                                                                images.shape,
                                                                masks.shape))

        # show 1 line of 'batch_size' images
        fig = plt.figure(figsize=(20, 20))
        for idx in np.arange(batch_size):
            ax = fig.add_subplot(1, batch_size, idx + 1, xticks=[], yticks=[])
            show_image(images[idx])
            ax.set_title("{}".format(fname[idx] if len(fname[idx]) < 15 else fname[idx][0:10] + "..."))

        if show_masks:
            # show 1 line of 'batch_size' masks
            fig = plt.figure(figsize=(20, 20))
            for idx in np.arange(batch_size):
                ax = fig.add_subplot(1, batch_size, idx + 1, xticks=[], yticks=[])
                show_image(masks[idx])

        cont_rows = cont_rows + 1
        if cont_rows >= row_limit:
            break