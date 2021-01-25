import math

import cv2
import openslide
import numpy as np
import skimage.io as sk_io
import skimage.color as sk_color
import skimage.filters as sk_filters
import skimage.morphology as sk_morphology


from sourcecode.logger_utils import *
from PIL import Image, ImageDraw, ImageFont, ImageOps
from openslide import OpenSlideError
from skimage import measure


BLACK = (0, 0, 0)
RED = (255, 0, 0)
LIME = (0, 255, 0)
BLUE = (0, 0 ,255)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
SILVER = (192, 192, 192)
GRAY = (128, 128, 128)
MAROON = (128, 0, 0)
OLIVE = (128, 128, 0)
GREEN = (0, 128, 0)
PURPLE = (128, 0, 128)
TEAL = (0, 128, 128)
NAVY = (0, 0, 128)
CRIMSON = (220, 20, 60)
GOLDEN_ROD = (218, 165, 32)
SIENNA = (160, 82, 45)
PINK = (255, 192, 203)
GREEN_YELLOW = (173, 255, 47)
BEIGE = (245,245,220)
ORANGE = (255, 165, 0)
AZURE = (240, 255, 255)
DODGER_BLUE = (30, 144, 255)
VIOLET = (238, 130, 238)
CHOCOLATE = (210, 105, 30)
TOMATO = (255, 99, 71)
LIGHT_GREEN = (144, 238, 144)
DARK_SEA_GREEN = (143, 188, 143)
GOLD = (255, 215, 0)
WHITE = (255, 255, 255)
COLOR_CLASSES = np.array([BLACK, RED, LIME, BLUE, YELLOW, CYAN, MAGENTA, SILVER, GRAY, MAROON, OLIVE, GREEN, PURPLE,
                          TEAL, NAVY, CRIMSON, GOLDEN_ROD, SIENNA, PINK, GREEN_YELLOW, BEIGE, ORANGE, AZURE,
                          DODGER_BLUE, VIOLET, CHOCOLATE, TOMATO, LIGHT_GREEN, DARK_SEA_GREEN, GOLD, WHITE])


GREEN_COLOR = (0, 255, 0)
YELLOW_COLOR = (255, 255, 0)
ORANGE_COLOR = (255, 165, 0)
RED_COLOR = (255, 0, 0)


MAGNIFICATION_SCALE = {
    "20.0": 1.0,
    "10.0": 2.0,
    "5.0": 4.0,
    "2.5": 8.0,
    "1.25": 16.0,
    "0.625": 32.0,
    "0.3125": 64.0,
    "0.15625": 128.0,
    "0.078125": 256.0
}


###
# OPEN SLIDE FUNCTIONS
###
def get_scale_by_magnification(magnification):
    return MAGNIFICATION_SCALE[str(magnification)]


def open_wsi(filename):
    """
    Open a whole-slide image (*.svs, etc).
    Args:
        filename: Name of the image file.
    Returns:
        An OpenSlide object representing a whole-slide image.
    """

    try:
        slide = openslide.open_slide(filename)
    except OpenSlideError:
        slide = None
    except FileNotFoundError:
        slide = None

    return slide


def scale_down_wsi(wsi_image, magnification, use_openslide_propeties=True):
    """
    Convert a WSI to a scaled-down PIL image.
    Args:
        wsi_image: Whole-slide image to be scaled down.
        magnification: Whole-slide image magnification to be used.
        use_openslide_propeties:
    Returns:
        Returns the scaled-down PIL image.
    """
    scale = get_scale_by_magnification(magnification)
    if use_openslide_propeties:
        level = wsi_image.level_downsamples.index(scale)
        new_dimension = wsi_image.level_dimensions[level]
    else:
        large_w, large_h = wsi_image.dimensions
        new_w = math.floor(large_w / scale)
        new_h = math.floor(large_h / scale)
        new_dimension = (new_w, new_h) if new_w > 100 else (new_w*2, new_h*2)

    return wsi_image.get_thumbnail(new_dimension)


def scale_down_camelyon16_img(image_file, magnification):

    # load image
    wsi_image = open_wsi(image_file)

    # scales down the image
    scale = get_scale_by_magnification(magnification)
    wsi_image_pil = scale_down_wsi(wsi_image, magnification)

    return wsi_image_pil, scale


def extract_normal_region_from_wsi(wsi_image_file, np_scaled_down_image, np_tumor_mask):

    logger.info("\t Extracting normal regions from wsi image: '{}'".format(wsi_image_file.split('/')[-1]))

    np_mask = tissue_mask(np_scaled_down_image)
    if np_tumor_mask is not None:
        np_mask[np_tumor_mask > 0] = 0

    np_masked_image = mask_rgb(np_scaled_down_image, np_mask)

    return np_mask, np_masked_image


def read_region(wsi_image_file, column, row, magnification=0.625, tile_size=20):

    # load image
    wsi_image = open_wsi(wsi_image_file)

    scale = get_scale_by_magnification(magnification)
    level = wsi_image.get_best_level_for_downsample(scale)

    tile_size_original = int(tile_size*scale)
    left = (column * tile_size_original)
    top = (row * tile_size_original)

    #print("Reading regions: {}x{} mag: {} scale: {} level: 0".format(tile_size_original, tile_size_original, magnification, scale))
    region_pil = wsi_image.read_region((left, top), 0, (tile_size_original, tile_size_original))
    region_np = np.asarray(region_pil)
    return region_pil, region_np[:, :, :3]


def draw_tile_border(draw, r_s, r_e, c_s, c_e, color=GREEN_COLOR, border_size=1, text=None):
    """
    Draw a border around a tile.
    Args:
        draw: Draw object for drawing on PIL image.
        r_s: Row starting pixel.
        r_e: Row ending pixel.
        c_s: Column starting pixel.
        c_e: Column ending pixel.
        color: RGB color of the border.
        border_size: Width of tile border in pixels.
        text: Label to draw into tile.
    """
    for x in range(0, border_size):
        draw.rectangle([(c_s + x, r_s + x), (c_e - 1 - x, r_e - 1 - x)], outline=color)

    if text is not None:
        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 15)
        (x, y) = draw.textsize(text, font)
        draw.text((c_s + 5, r_s + 5), text, (255, 255, 255), font=font)


def draw_heat_grid(np_processed_img, tile_size):

    shape = np_processed_img.shape
    heat_grid = []
    tile_position = 0
    pil_processed_img = np_to_pil(np_processed_img)
    draw = ImageDraw.Draw(pil_processed_img)

    for height in range(0, shape[0], tile_size):
        for width in range(0, shape[1], tile_size):

            row = int(height / tile_size)
            column = int(width / tile_size)

            r_s = row * tile_size
            r_e = r_s + tile_size
            c_s = column * tile_size
            c_e = c_s + tile_size

            cropped_np_img = np_processed_img[r_s:r_e, c_s:c_e]
            tissue_area = tissue_percent(cropped_np_img)
            #            print("tile: {} - {}% r{} c{}".format(tile_position, tissue_area, row, column))

            if tissue_area <= 5.0:
                color = GREEN_COLOR
            elif 5.0 < tissue_area <= 10.0:
                color = YELLOW_COLOR
            elif 10.0 < tissue_area <= 80.0:
                color = ORANGE_COLOR
            else:
                color = RED_COLOR

            label = None
            if height == 0:
                label = str(int(width / tile_size))
            elif width == 0:
                label = str(int(height / tile_size))

            tile_position += 1
            location = (c_s, r_s)
            size = (tile_size, tile_size)
            tile = (tile_position, row, column, location, size, color)

            heat_grid.append(tile)
            draw_tile_border(draw, r_s, r_e, c_s, c_e, color, text=label)

    return pil_processed_img, heat_grid, tile_position
###


def extract_tumor_region_from_wsi(contours, wsi_image_file, magnification):

    logger.info("\t Extracting tumor regions from wsi image: '{}'".format(wsi_image_file.split('/')[-1]))

    wsi_image_pil, scale = scale_down_camelyon16_img(wsi_image_file, magnification)
    np_scaled_down_image = pil_to_np(wsi_image_pil)

    # find the tumor mask
    pil_mask = np_to_pil(np.zeros((np_scaled_down_image.shape[0], np_scaled_down_image.shape[1]), dtype=np.uint8))
    draw = ImageDraw.Draw(pil_mask)

    region_label = 1
    for idx, (region_name, annotation_type, group, color, points) in enumerate(contours):
        if group != "_2" and group != "Exclusion" and len(points) > 1:
            points_scaled_down = [tuple(pt * (1 / scale) for pt in p) for p in points]
            draw.polygon(points_scaled_down, outline=None, fill=region_label)
            region_label += 1
    for idx, (region_name, annotation_type, group, color, points) in enumerate(contours):
        if group == "_2" or group == "Exclusion" and len(points) > 1:
            points_scaled_down = [tuple(pt * (1 / scale) for pt in p) for p in points]
            draw.polygon(points_scaled_down, outline=None, fill=0)

    np_regions_label = pil_to_np(pil_mask).astype(np.uint8)
    np_mask = np_regions_label.astype(bool)
    #
    #np_mask = np_regions_label > 0
    np_regions_label = measure.label(np_mask, connectivity=2)
    #
    np_masked_image = mask_rgb(np_scaled_down_image, np_mask)

    return np_scaled_down_image, np_regions_label, np_mask, np_masked_image


def load_np_image(path, color_model="RGB"):

    pil_img = load_pil_image(path, gray=color_model == "GRAY", color_model=color_model)
    return pil_to_np(pil_img)


def load_pil_image(path, gray=False, color_model="RGB"):

    with open(path, 'rb') as f:

        if gray:
            return Image.open(f).convert('L')     # grayscale

        elif color_model == "HSV":
            # For HSV, 'H' range is [0, 179], 'S' range is [0, 255] and 'V' range is [0, 255]
            return Image.open(f).convert('HSV')      # hsv

        elif color_model == "LAB":
            rgb = sk_io.imread(path)
            if rgb.shape[2] > 3:  # removes the alpha channel
                rgb = sk_color.rgba2rgb(rgb)

            lab = sk_color.rgb2lab(rgb)
            # For LAB, 'L' range is [0, 100], 'A' range is [-127, 127] and 'B' range is [-127, 127]
            lab_scaled = ((lab + [0, 128, 128]) / [100, 255, 255])*255
            return Image.fromarray(lab_scaled.astype(np.uint8))

        return Image.open(f).convert('RGB')    # rgb


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), int(height))

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (int(width), int(h * r))

    #print("(h,w): {} / dim: {}".format((h,w), dim))
    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def pil_to_np(pil_img):
    """
    Convert a PIL Image to a NumPy array.
    Note that RGB PIL (w, h) -> NumPy (h, w, 3).
    Args:
    pil_img: The PIL Image.
    Returns:
    The PIL image converted to a NumPy array.
    """

    rgb = np.asarray(pil_img)
    return rgb


def np_to_pil(np_img):
    """
    Convert a NumPy array to a PIL Image.
    Args:
        np_img: The image represented as a NumPy array.
    Returns:
    The NumPy array converted to a PIL Image.
    """

    if np_img.dtype == "bool":
        np_img = np_img.astype("uint8") * 255
    elif np_img.dtype == "float64":
        np_img = (np_img * 255).astype("uint8")

    return Image.fromarray(np_img)


def rgb_to_hsv(np_img):
    """
    Filter RGB channels to HSV (Hue, Saturation, Value).
    Args:
        np_img: RGB image as a NumPy array.
    Returns:
        Image as NumPy array in HSV representation.
    """

    return sk_color.rgb2hsv(np_img)


def rgb_to_lab(np_img):
    """
    Filter RGB channels to CIE L*a*b*.
    Args:
        np_img: RGB image as a NumPy array.
    Returns:
        Image as NumPy array in Lab representation.
    """

    if np_img.shape[2] > 3:  # removes the alpha channel
        np_img = sk_color.rgba2rgb(np_img)

    lab = sk_color.rgb2lab(np_img)
    # For LAB, 'L' range is [0, 100], 'A' range is [-127, 127] and 'B' range is [-127, 127]
    lab = ((lab + [0, 128, 128]) / [100, 255, 255])
    return lab


def lab_to_rgb(np_img):
    """
    Filter LAB channels to RGB (Red, Green, Blue).
    Args:
        np_img: LAB image as a NumPy array.
    Returns:
        Image as NumPy array in RGB representation.
    """

    lab_rescaled = ((np_img - [0, 128, 128]) * [100, 255, 255])/255
    rgb = sk_color.lab2rgb(lab_rescaled)
    return rgb


def hsv_to_rgb(np_img):
    """
    Filter HSV channels to RGB (Red, Green, Blue).
    Args:
        np_img: HSV image as a NumPy array.
    Returns:
        Image as NumPy array in RGB representation.
    """

    return sk_color.hsv2rgb(np_img)


def filter_purple_pink(np_img, output_type="bool"):
    """
    Create a mask to filter out pixels where the values are similar to purple and pink.
    Args:
        np_img: RGB image as a NumPy array.
        output_type: Type of array to return (bool, float, or uint8).
    Returns:
        NumPy array representing a mask where pixels with purple/pink values have been masked out.
    """

    np_img_bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(np_img_bgr, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (130, 30, 50), (170, 255, 255))
    mask = basic_threshold(mask, threshold=0, output_type="bool")

    return parse_output_type(mask, output_type)


def remove_small_objects(np_img, min_size=3000, output_type="bool"):
    """
    Filter image to remove small objects (connected components) less than a particular minimum size.
    Args:
        np_img: Image as a NumPy array of type bool.
        min_size: Minimum size of small object to remove.
        output_type: Type of array to return (bool, float, or uint8).
    Returns:
        NumPy array (bool, float, or uint8).
    """

    result = np_img.astype(bool)  # make sure mask is boolean
    result = sk_morphology.remove_small_objects(result, min_size=min_size)
    return parse_output_type(result, output_type)


def fill_small_holes(np_img, area_threshold=3000, output_type="bool"):
    """
    Filter image to remove small holes less than a particular size.
    Args:
        np_img: Image as a NumPy array of type bool.
        area_threshold: Remove small holes below this area.
        output_type: Type of array to return (bool, float, or uint8).
    Returns:
        NumPy array (bool, float, or uint8).
    """

    result = sk_morphology.remove_small_holes(np_img, area_threshold=area_threshold)
    return parse_output_type(result, output_type)


def tissue_mask(np_img):

    # To prevent selecting background patches, slides are converted to HSV, blurred,
    # and patches filtered out if maximum pixel saturation lies below 0.07
    # (which was validated to not throw out tumor data in the training set).

    np_tissue_mask = filter_purple_pink(np_img)
    np_tissue_mask = fill_small_holes(np_tissue_mask, area_threshold=3000)
    np_tissue_mask = remove_small_objects(np_tissue_mask, min_size=3000)
    return np_tissue_mask


def mask_rgb(rgb, mask):
    """
    Apply a binary (T/F, 1/0) mask to a 3-channel RGB image and output the result.
    Args:
        rgb: RGB image as a NumPy array.
        mask: An image mask to determine which pixels in the original image should be displayed.
    Returns:
        NumPy array representing an RGB image with mask applied.
    """

    result = rgb * np.dstack([mask, mask, mask])
    return result


def blend_image(image, mask, foreground='red', alpha=0.3, inverse=False):

    if inverse:
        mask = ImageOps.invert(mask)

    foreground = Image.new('RGB', image.size, color=foreground)
    composite = Image.composite(image, foreground, mask)
    return Image.blend(image, composite, alpha)


def mask_percent(np_img):
    """
    Determine the percentage of a NumPy array that is masked (how many of the values are 0 values).
    Args:
        np_img: Image as a NumPy array.
    Returns:
        The percentage of the NumPy array that is masked.
    """

    if (len(np_img.shape) == 3) and (np_img.shape[2] == 3):
        np_sum = np_img[:, :, 0] + np_img[:, :, 1] + np_img[:, :, 2]
        mask_percentage = 0 if np_sum.size == 0 else 100 - np.count_nonzero(np_sum) / np_sum.size * 100
    else:
        mask_percentage = 0 if np_img.size == 0 else 100 - np.count_nonzero(np_img) / np_img.size * 100

    return mask_percentage


def tissue_percent(np_img):
    """
    Determine the percentage of a NumPy array that is tissue (not masked).
    Args:
        np_img: Image as a NumPy array.
    Returns:
        The percentage of the NumPy array that is tissue.
    """

    return 100 - mask_percent(np_img)


def basic_threshold(np_img, threshold=0.0, output_type="bool"):
    """
    Return mask where a pixel has a value if it exceeds the threshold value.
    Args:
        np_img: Binary image as a NumPy array.
        threshold: The threshold value to exceed.
        output_type: Type of array to return (bool, float, or uint8).
    Returns:
        NumPy array representing a mask where a pixel has a value (T, 1.0, or 255) if the corresponding input array pixel exceeds the threshold value.
    """

    result = (np_img > threshold)
    return parse_output_type(result, output_type)


def otsu_threshold(np_img, output_type="bool"):
    """
    Compute Otsu threshold on image as a NumPy array and return binary image based on pixels above threshold.
    Args:
        np_img: Image as a NumPy array.
        output_type: Type of array to return (bool, float, or uint8).
    Returns:
        NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a pixel above Otsu threshold.
    """

    otsu_thresh_value = sk_filters.threshold_otsu(np_img)
    result = (np_img > otsu_thresh_value)
    return parse_output_type(result, output_type)


def parse_output_type(np_img, output_type="bool"):
    if output_type == "bool":
        pass
    elif output_type == "float":
        np_img = np_img.astype(float)
    else:
        np_img = np_img.astype("uint8") * 255

    return np_img


def output_map_to_rgb_image(output_map):

    r = np.zeros_like(output_map).astype(np.uint8)
    g = np.zeros_like(output_map).astype(np.uint8)
    b = np.zeros_like(output_map).astype(np.uint8)

    colors = np.copy(COLOR_CLASSES)
    qtd = len(np.unique(output_map)) - len(colors)
    if qtd > 0:
        colors = np.append(colors, COLOR_CLASSES[1:qtd], axis=0)
    else:
        colors = colors[0:len(COLOR_CLASSES)]

    for cls in range(0, len(colors)):
        idx = output_map == cls
        r[idx] = colors[cls, 0]
        g[idx] = colors[cls, 1]
        b[idx] = colors[cls, 2]
        rgb = np.stack([r, g, b], axis=2)

    return rgb


def show_np_img(np_img, text=None):
    """
    Convert a NumPy array to a PIL image, add text to the image, and display the image.
    Args:
        np_img: Image as a NumPy array.
        text: The text to be added to the image.
    """

    pil_img = np_to_pil(np_img)
    show_pil_img(pil_img, text)


def show_pil_img(pil_img, text=None):
    """
    Add text to the image, and display the image.
    Args:
        pil_img: PIL Image.
        text: The text to be added to the image.
    """

    # if gray, convert to RGB for display
    if pil_img.mode == 'L':
        pil_img = pil_img.convert('RGB')

    if text is not None:
        draw = ImageDraw.Draw(pil_img)
        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 48)
        (x, y) = draw.textsize(text, font)
        draw.rectangle([(0, 0), (x + 5, y + 4)], fill=(0, 0, 0), outline=(0, 0, 0))
        draw.text((2, 0), text, (255, 0, 0), font=font)

    pil_img.show()