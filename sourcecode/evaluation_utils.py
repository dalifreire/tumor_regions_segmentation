import os
import csv

from sourcecode.wsi_image_utils import *
from sklearn import metrics


def camelyon16_ground_truth(target_csv_file, exclusions=None):
    ground_truth = []
    with open(target_csv_file) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if exclusions is None or row[0] not in exclusions:
                ground_truth.append(row[1])
    return ground_truth


def camelyon16_wsi_classification_results(results_folder, exclusions=None):
    pred = []
    for file in sorted(os.listdir(results_folder)):
        if exclusions is None or file not in exclusions:
            full_path = os.path.join(results_folder, file)
            pred.append("Tumor" if os.path.getsize(full_path) > 0 and camelyon16_check_prob(full_path) else "Normal")
    return pred


def camelyon16_check_prob(prob_file, threshold=0.0):
    with open(prob_file) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if float(row[0]) >= threshold:
                return True
    return False


def image_path_to_np(image_path):

    image = load_pil_image(image_path, gray=True)
    image_np = np.asarray(image)
    return image_np


def accuracy_score(target, prediction, pixel=True):

    if not pixel or str(target).lower().endswith(".csv"):
        total_patches, tn, fp, fn, tp = tn_fp_fn_tp(target, prediction, False)
        return (tp + tn) / (tp + tn + fp + fn)

    if type(target).__module__ == np.__name__:
        mask_np = np.copy(target)
        output_np = np.copy(prediction)
    else:
        mask_np = image_path_to_np(target)
        output_np = image_path_to_np(prediction)

    return metrics.accuracy_score(mask_np.ravel(), output_np.ravel())


def roc_curve(target, prediction, pixel=True):

    if str(target).lower().endswith(".csv"):

        ground_truth = [0 if x == "Normal" else 1 for x in camelyon16_ground_truth(target)]
        classification_results = [0 if x == "Normal" else 1 for x in camelyon16_wsi_classification_results(prediction)]
        fpr, tpr, thresholds = metrics.roc_curve(ground_truth, classification_results)
        return fpr, tpr, thresholds

    if type(target).__module__ == np.__name__:
        mask_np = np.copy(target)
        output_np = np.copy(prediction)
    else:
        mask_np = image_path_to_np(target)
        output_np = image_path_to_np(prediction)

    return metrics.roc_curve(mask_np.ravel(), output_np.ravel(), average='micro')


def roc_auc_score(target, prediction, pixel=True):

    if str(target).lower().endswith(".csv"):

        ground_truth = [0 if x == "Normal" else 1 for x in camelyon16_ground_truth(target)]
        classification_results = [0 if x == "Normal" else 1 for x in camelyon16_wsi_classification_results(prediction)]
        return metrics.roc_auc_score(ground_truth, classification_results)

    if type(target).__module__ == np.__name__:
        mask_np = np.copy(target)
        output_np = np.copy(prediction)
    else:
        mask_np = image_path_to_np(target)
        output_np = image_path_to_np(prediction)

    total_patches, tn, fp, fn, tp = tn_fp_fn_tp(target, prediction, pixel)
    return 0.0 if tp == 0 else metrics.roc_auc_score(mask_np.ravel(), output_np.ravel(), average='micro')


def precision_score(target, prediction, pixel=True):

    if not pixel or str(target).lower().endswith(".csv"):
        total_patches, tn, fp, fn, tp = tn_fp_fn_tp(target, prediction, pixel=False)
        return tp / (tp + fp)

    if type(target).__module__ == np.__name__:
        mask_np = np.copy(target)
        output_np = np.copy(prediction)
    else:
        mask_np = image_path_to_np(target)
        output_np = image_path_to_np(prediction)

    return metrics.precision_score(mask_np, output_np, average='micro')


def recall_score(target, prediction, pixel=True):

    if not pixel or str(target).lower().endswith(".csv"):
        return sensitivity_score(target, prediction, False)

    if type(target).__module__ == np.__name__:
        mask_np = np.copy(target)
        output_np = np.copy(prediction)
    else:
        mask_np = image_path_to_np(target)
        output_np = image_path_to_np(prediction)

    return metrics.recall_score(mask_np, output_np, average='micro')


def f1_score(target, prediction, pixel=True):

    if not pixel or str(target).lower().endswith(".csv"):
        return dice_score(target, prediction, False)

    if type(target).__module__ == np.__name__:
        mask_np = np.copy(target)
        output_np = np.copy(prediction)
    else:
        mask_np = image_path_to_np(target)
        output_np = image_path_to_np(prediction)

    return metrics.f1_score(mask_np, output_np, average='micro')


def sensitivity_score(target, prediction, pixel=True):

    total_patches, tn, fp, fn, tp = tn_fp_fn_tp(target, prediction, pixel)
    return tp/(tp + fn)


def specificity_score(target, prediction, pixel=True):

    total_patches, tn, fp, fn, tp = tn_fp_fn_tp(target, prediction, pixel)
    return 0.0 if (tn + fp) == 0 else tn/(tn + fp)


def dice_score(target, prediction, pixel=True):

    total_patches, tn, fp, fn, tp = tn_fp_fn_tp(target, prediction, pixel)
    # 2 * (precision * recall) / (precision + recall)
    return (2*tp)/(2*tp + fp + fn)


def jaccard_score(target, prediction, pixel=True):

    if not pixel or str(target).lower().endswith(".csv"):
        total_patches, tn, fp, fn, tp = tn_fp_fn_tp(target, prediction, pixel=False)
        return tp / (tp + fp + fn)

    if type(target).__module__ == np.__name__:
        mask_np = np.copy(target)
        output_np = np.copy(prediction)
    else:
        mask_np = image_path_to_np(target)
        output_np = image_path_to_np(prediction)

    return metrics.jaccard_score(mask_np, output_np, average='micro')


def tn_fp_fn_tp(target, prediction, pixel=True):

    if str(target).lower().endswith(".csv"):

        normal_images = set()
        tumor_images = set()
        with open(target) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                if row[1] == "Normal":
                    normal_images.add(row[0])
                elif row[1] == "Tumor":
                    tumor_images.add(row[0])

        tumor_images_predicted = set()
        for file in sorted(os.listdir(prediction)):
            #if os.path.getsize(os.path.join(prediction, file)) > 0:
            full_path = os.path.join(prediction, file)
            if os.path.getsize(full_path) > 0 and camelyon16_check_prob(full_path):
                tumor_images_predicted.add(file.replace(".csv", ""))

        total = len(normal_images) + len(tumor_images)
        tp = len(tumor_images & tumor_images_predicted)
        fp = len(normal_images & tumor_images_predicted)
        tn = len(normal_images - tumor_images_predicted)
        fn = len(tumor_images - tumor_images_predicted)
        return total, tn, fp, fn, tp

    if not pixel:

        non_roi_dir = "{}/1-normal".format(target)
        roi_dir = "{}/2-tumor".format(target)

        roi_files = set([os.path.basename(os.path.normpath(x)) for x in os.listdir(roi_dir)])
        non_roi_files = set([os.path.basename(os.path.normpath(x)) for x in os.listdir(non_roi_dir)])
        roi_files_predicted = set([os.path.basename(os.path.normpath(x)) for x in os.listdir(prediction)])

        total_patches = len(roi_files) + len(non_roi_files)
        tp = len(roi_files & roi_files_predicted)
        fp = len(non_roi_files & roi_files_predicted)
        tn = len(non_roi_files - roi_files_predicted)
        fn = len(roi_files - roi_files_predicted)
        return total_patches, tn, fp, fn, tp

    if type(target).__module__ == np.__name__:

        mask_np = np.copy(target)
        output_np = np.copy(prediction)

    else:

        mask_np = image_path_to_np(target)
        output_np = image_path_to_np(prediction)

    tn, fp, fn, tp = metrics.confusion_matrix(mask_np.ravel(), output_np.ravel(), labels=[0, 1]).ravel()
    return (tn+fp+fn+tp), tn, fp, fn, tp
