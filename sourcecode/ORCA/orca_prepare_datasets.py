import os
import sys

current_path = os.path.abspath('.')
root_path = os.path.dirname(os.path.dirname(current_path))
sys.path.append(root_path)

from sourcecode.ORCA.orca_dataloader import *



magnification=0.625
scale = get_scale_by_magnification(magnification)
tile_size=20
tile_size_original = int(scale * tile_size)


cls = "tumor"
dataset_types = ["training", "testing"]
for dataset_type in sorted(dataset_types):
    
    logger.info("{} images".format(dataset_type))
    
    orca_dir = "../../datasets/ORCA"
    annotations_dir = "{}/{}/lesion_annotations".format(orca_dir, dataset_type)
    wsi_images_dir = "{}/{}/{}/wsi".format(orca_dir, dataset_type, cls)
    patch_images_dir = "{}/{}/{}/patch/640x640".format(orca_dir, dataset_type, cls)
    
    for r, d, f in sorted(os.walk(wsi_images_dir)):
        for wsi_file in sorted(f):
            
            wsi_image_file = "{}/{}".format(r, wsi_file)
            wsi_image_number = wsi_file.replace(".png", "")
            
            file_is_png = wsi_image_file.lower().endswith('.png')
            if file_is_png:

                logger.info("Processing tma '{}'".format(wsi_file))
                
                # check directory to save image-patches
                dir_to_save = "{}/{}".format(patch_images_dir, wsi_image_number)
                if not os.path.exists(dir_to_save):
                    os.makedirs("{}/01-roi/01-original".format(dir_to_save))
                    os.makedirs("{}/01-roi/02-mask".format(dir_to_save))
                    os.makedirs("{}/01-roi/03-roi".format(dir_to_save))             
                    os.makedirs("{}/02-non_roi/01-original".format(dir_to_save))
                    os.makedirs("{}/02-non_roi/02-mask".format(dir_to_save))
                
                # tumor annotations mask
                annotation_file = "{}/{}_mask.png".format(annotations_dir, wsi_image_number)
                wsi_mask_pil = load_pil_image(annotation_file, gray=True)
                
                # tumor tissue region
                wsi_image = open_wsi(wsi_image_file)
                max_w, max_h = wsi_image.dimensions
                wsi_image_pil = load_pil_image(wsi_image_file, gray=False)
                pil_scaled_down_image = scale_down_wsi(wsi_image, magnification, False)
                
                np_tumor_mask = np.zeros((wsi_mask_pil.size[0], wsi_mask_pil.size[1]), dtype=bool)
                np_tumor_mask[pil_to_np(wsi_mask_pil) == 255] = True
                pil_tumor_mask = np_to_pil(np_tumor_mask)
                
                np_normal_mask = np.zeros((wsi_mask_pil.size[0], wsi_mask_pil.size[1]), dtype=bool)
                np_normal_mask[pil_to_np(wsi_mask_pil) < 255] = True
                np_normal_mask[pil_to_np(wsi_mask_pil) == 0] = False
                pil_normal_mask = np_to_pil(np_normal_mask)
                
                pil_background = np_to_pil(np_normal_mask | np_tumor_mask)

                wsi_image_np = pil_to_np(wsi_image_pil)
                np_tumor_masked = mask_rgb(wsi_image_np, np_tumor_mask)
                np_tumor_masked = image_resize(np_tumor_masked, height=pil_scaled_down_image.size[1])
                pil_tumor_masked = np_to_pil(np_tumor_masked)
                
                np_normal_masked = mask_rgb(wsi_image_np, np_normal_mask)
                np_normal_masked = image_resize(np_normal_masked, height=pil_scaled_down_image.size[1])
                pil_normal_masked = np_to_pil(np_normal_masked)

                blend_tumor = blend_image(wsi_image_pil, pil_tumor_mask, foreground='red', inverse=True)
                blend_tissue = blend_image(blend_tumor, pil_normal_mask, foreground='green', inverse=True)
                blend_background = blend_image(blend_tissue, pil_background, foreground='blue', inverse=False).resize(pil_tumor_masked.size, Image.ANTIALIAS)

                pil_scaled_down_image.save('{}/{}_1.png'.format(dir_to_save, wsi_image_number))
                blend_background.save('{}/{}_2.png'.format(dir_to_save, wsi_image_number))
                pil_normal_masked.save('{}/{}_3.png'.format(dir_to_save, wsi_image_number))
                pil_tumor_masked.save('{}/{}_4.png'.format(dir_to_save, wsi_image_number))
                
                #heat grid normal
                pil_img_normal_result, heat_grid_normal, number_of_tiles_normal = draw_heat_grid(np_normal_masked, tile_size)
                pil_img_normal_result.save('{}/{}_5.png'.format(dir_to_save, wsi_image_number))

                # heat grid tumor
                pil_img_result, heat_grid_tumor, number_of_tiles = draw_heat_grid(np_tumor_masked, tile_size)
                pil_img_result.save('{}/{}_6.png'.format(dir_to_save, wsi_image_number))
                
                used_patches = set()
                tiles_tumor_tissue = []
                tiles_normal_tissue = []
                tiles_background = []
                for idx, (position, row, column, location, size, color) in enumerate(heat_grid_tumor):

                    tile = (position, row, column, location, size, color)                    
                    if color == YELLOW_COLOR or color == ORANGE_COLOR or color == RED_COLOR:
                        tiles_tumor_tissue.append(tile)
                        used_patches.add("r{}c{}".format(row, column))
                
                for idx, (position, row, column, location, size, color) in enumerate(heat_grid_normal):

                    name = "r{}c{}".format(row, column)
                    tile = (position, row, column, location, size, color)
                    if name not in used_patches:
                        if color == GREEN_COLOR:
                            tiles_background.append(tile)
                        else:
                            tiles_normal_tissue.append(tile)
                
                logger.info("\t {} patches of 640x640 size.".format(len(heat_grid_normal)))
                logger.info("\t\t {} patches of 640x640 (normal tissue).".format(len(tiles_normal_tissue)))
                logger.info("\t\t {} patches of 640x640 (tumor tissue).".format(len(tiles_tumor_tissue)))
                logger.info("\t\t {} patches of 640x640 (background).".format(len(tiles_background)))
                
                
                # extract and save normal patches
                for (position, row, column, location, size, color) in tiles_normal_tissue:
                    
                    r_s = row * tile_size_original
                    r_e = r_s + (tile_size_original if (r_s + tile_size_original) <= max_w else (max_w - r_s))
                    c_s = column * tile_size_original
                    c_e = c_s + (tile_size_original if (c_s + tile_size_original) <= max_h else (max_h - c_s))
                    
                    np_tile_mask = np.zeros((tile_size_original, tile_size_original), dtype=bool)
                    np_tile_mask[0:(r_e-r_s), 0:(c_e-c_s)] = np_tumor_mask[r_s:r_e, c_s:c_e]
                    
                    # only tile with valid size
                    #if np_tile_mask.shape[0] == tile_size_original and np_tile_mask.shape[1] == tile_size_original:

                    tile_pil, tile_np = read_region(wsi_image_file, column, row, magnification, tile_size)
                    left = (column * tile_size_original)
                    top = (row * tile_size_original)

                    pil_mask = np_to_pil(np.zeros((tile_np.shape[0], tile_np.shape[1]), dtype=np.uint8))
                    np_tile_mask = pil_to_np(pil_mask).astype(bool)

                    # save the extracted normal tile
                    tile_pil.save('{}/02-non_roi/{}/{}_r{}c{}.png'.format(dir_to_save, "01-original", wsi_image_number, row, column))
                    np_to_pil(np_tile_mask).save('{}/02-non_roi/{}/{}_r{}c{}.png'.format(dir_to_save, "02-mask", wsi_image_number, row, column))

                # extract and save tumor patches
                for (position, row, column, location, size, color) in tiles_tumor_tissue:
                    
                    r_s = row * tile_size_original
                    r_e = r_s + (tile_size_original if (r_s + tile_size_original) <= max_w else (max_w - r_s))
                    c_s = column * tile_size_original
                    c_e = c_s + (tile_size_original if (c_s + tile_size_original) <= max_h else (max_h - c_s))
                    
                    np_tile_mask = np.zeros((tile_size_original, tile_size_original), dtype=bool)
                    np_tile_mask[0:(r_e-r_s), 0:(c_e-c_s)] = np_tumor_mask[r_s:r_e, c_s:c_e]
                    
                    # only tile with valid size
                    #if np_tile_mask.shape[0] == tile_size_original and np_tile_mask.shape[1] == tile_size_original:

                    tile_pil, tile_np = read_region(wsi_image_file, column, row, magnification, tile_size)
                    left = (column * tile_size_original)
                    top = (row * tile_size_original)
                    
                    pil_tile_roi = blend_image(np_to_pil(tile_np), np_to_pil(np_tile_mask), foreground='blue', inverse=True)
                    #pil_tile_roi = blend_image(pil_tile_roi, np_to_pil(np_tile_mask), foreground='green', inverse=False)

                     # save the extracted tumor image-patch
                    tile_pil.save('{}/01-roi/{}/{}_r{}c{}.png'.format(dir_to_save, "01-original", wsi_image_number, row, column))
                    np_to_pil(np_tile_mask).save('{}/01-roi/{}/{}_r{}c{}.png'.format(dir_to_save, "02-mask", wsi_image_number, row, column))
                    pil_tile_roi.save('{}/01-roi/{}/{}_r{}c{}.png'.format(dir_to_save, "03-roi", wsi_image_number, row, column))
                







magnification=0.625
scale = get_scale_by_magnification(magnification)
tile_size=20
tile_size_original = int(scale * tile_size)


cls = "tumor"
dataset_types = ["training", "testing"]
for dataset_type in sorted(dataset_types):
    
    logger.info("{} images".format(dataset_type))
    
    orca_dir = "../../datasets/ORCA"
    annotations_dir = "{}/{}/lesion_annotations".format(orca_dir, dataset_type)
    tma_images_dir = "{}/{}/{}/wsi".format(orca_dir, dataset_type, cls)
    
    orca_512x512_dir = "../../datasets/ORCA_512x512"
    resized_annotations_dir = "{}/{}/lesion_annotations".format(orca_512x512_dir, dataset_type)
    resized_tma_images_dir = "{}/{}/{}/tma".format(orca_512x512_dir, dataset_type, cls)
    
    for r, d, f in sorted(os.walk(tma_images_dir)):
        for tma_file in sorted(f):
            
            tma_image_file = "{}/{}".format(r, tma_file)
            tma_image_number = tma_file.replace(".png", "")
            
            file_is_png = tma_image_file.lower().endswith('.png')
            if file_is_png:

                logger.info("Processing tma '{}'".format(tma_file))
                
                # tumor annotations mask
                annotation_file = "{}/{}_mask.png".format(annotations_dir, tma_image_number)
                tma_mask_pil = load_pil_image(annotation_file, gray=True)
                
                # tumor tma image
                tma_image_pil = load_pil_image(tma_image_file, gray=False)
                
                #print("{}/{}_mask.png".format(resized_annotations_dir, tma_image_number))
                resized_tma_mask_np = image_resize(pil_to_np(tma_mask_pil), width=512, height=512)
                np_tumor_mask = np.zeros((512, 512), dtype=bool)
                np_tumor_mask[resized_tma_mask_np == 255] = True
                pil_tumor_mask = np_to_pil(np_tumor_mask)
                pil_tumor_mask.save("{}/{}_mask.png".format(resized_annotations_dir, tma_image_number))
                
                #print("{}/{}".format(resized_tma_images_dir, tma_file))
                resized_tma_image_np = image_resize(pil_to_np(tma_image_pil), width=512, height=512)
                np_to_pil(resized_tma_image_np).save("{}/{}".format(resized_tma_images_dir, tma_file))
                