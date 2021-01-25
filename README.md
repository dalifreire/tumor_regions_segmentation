# WSI tumor regions segmentation

Automated Detection of Tumor Regions from H&amp;E-stained Whole Slide Images Using Fully Convolutional Neural Networks.

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#whole-slide-images">Whole Slide Images</a></li>
    <li><a href="#proposed-method">Proposed method</a></li>
    <li>
        <a href="#datasets">Datasets</a>
        <ul>
            <li><a href="#ocdc">OCDC</a></li>
            <li><a href="#camelyon16">CAMELYON16</a></li>
            <li><a href="#orca">ORCA</a></li>
        </ul>
    </li>
    <li><a href="#sourcecode">Sourcecode</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>


# Whole slide images

Whole slide Images (WSI) are multi-gigapixel images created from tissue sections that can contain many different types of cells and regions – such as keratin, lymphocytes, blood vessels, glands, muscle and tumor cells. WSIs may also contain some derived artifacts from the image acquisition process, like tissue folding and blurring, that need to be ignored. 

A single H&E-stained histological WSI, as shown below, may contain thousands of nuclei and other cell structures that should be analyzed by pathologists at diagnostic time. This large amount of information requires very skilled professionals and a great effort to label the relevant structures to create these image datasets.

![](images/wsi.svg)


# Proposed method

The proposed method for localization and segmentation of oral squamous cell carcinoma (OSCC) tumor region-derived H&E-stained WSI is established on four pivotal steps: 
1. Scales down the WSI at a factor of 32× to perform background removal and tissue detection; 
2. Image-patch extraction from the identified tissue regions; 
3. Pixel-level cancer probability computation for each image-patch using FCN; 
4. Tumor regions segmentation by thresholding the final probability image;
 
![](images/method-overview.svg)

We propose to use a fully convolutional neural network (FCN) to address the challenge of patch-based segmentation of tumor regions in oral cavity-derived H&E-stained WSI at pixel level. The proposed FCN architecture, which consists of a couple of encoding (down) and decoding (up) layers, is based on [U-Net model](http://lmb.informatik.uni-freiburg.de/Publications/2015/RFB15a) and is illustrated below. 

![](images/proposed-network-architecture.svg)


# Datasets

## OCDC

Our oral cavity-derived cancer whole slide images dataset (OCDC) was built using tissue specimens collected from human patients diagnosed with OSCC. All OSCC cases were retrieved from the Department of Oral and Medical Pathology Services archives between 2006 and 2013 under approval by the Committee on Research and Ethics of our Institution (CAAE number: 15188713.9.0000.5152).

A total of 15 whole slide images were digitized using the Slide Scanner Aperio AT2 (Leica Biosystems Imaging, Inc., Nussloch, Germany) coupled to a computer (Dell Precision T3600) at 20× magnification and pixel-level resolution of 0.5025 μm × 0.5025 μm. The digitized images have different sizes – the larger one has almost three billions of pixels (63,743×45,472 pixels) – and were stored in SVS format using the RGB (red, green, blue) color model. A total of 1,050 image-patches of size 640×640 pixels were randomly extracted from these 15 WSI and the content of each image-patch was hand-annotated by a well-trained pathologist. The image-patch dataset was split into two subsets: the training set with 840 image-patches and the test set with 210 image-patches.

Details about the ODCD dataset are available at [datasets/OCDC folder](datasets/OCDC).

## CAMELYON16
The [CAMELYON16 dataset](https://camelyon16.grand-challenge.org) is a publicly available dataset with a combination of 399 WSIs of sentinel lymph node with breast cancer metastases tissue sections collected from two medical centers in the Netherlands for the [CAMELYON16 challenge](https://doi.org/10.1001/jama.2017.14585). These 399 WSIs were split into 270 for training and 129 for testing. The ground truth data for training and testing were provided as XML files containing the vertices of the delineation of tumor regions at WSI level. 

Details about the CAMELYON16 dataset are available at [datasets/CAMELYON16 folder](datasets/CAMELYON16).


## ORCA
The [ORal Cancer Annotated dataset (ORCA)](https://sites.google.com/unibas.it/orca) is a collection of 200 OSCC annotated images derived from the [Cancer Genome Atlas (TCGA) dataset](https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga). 

Details about the ORCA dataset are available at [datasets/ORCA folder](datasets/ORCA).


# Source code

The proposed method and the experimental evaluation were implemented using the [PyTorch framework](https://pytorch.org/get-started/locally/). The FCN model was trained using a desktop computer (Intel Core i7 3.4GHz×8 processor, 32 GB memory, 1TB SSD) equipped with GeForce GTX 1050 Ti graphic card and Ubuntu 20.04 operational system. For OCDC dataset the elapsed time to train the model during 500 epochs was about five days using 840 images of size 640×640 pixels. The elapsed time to train the CAMELYON16 dataset during 10 epochs using 109,278 images was about 15 days. After training, the elapsed time to process an 640×640 pixels input image-patch was about 0.8 seconds.

To provide better understanding and make this work as reproducible as possible, the source code is publicly available at [source code folder](sourcecode).




# Contact

Dalí F. D. dos Santos (dalifreire@gmail.com)

<!---
# Publications

[1](https://ieeexplore.ieee.org/document/9145157) D. F. D. dos Santos, T. A. A. Tosta, A. B. Silva, P. R. de Faria, B. A. N. Travençolo and M. Z. do Nascimento, "Automated Nuclei Segmentation on Dysplastic Oral Tissues Using CNN," 2020 International Conference on Systems, Signals and Image Processing (IWSSIP), Niterói, Brazil, 2020, pp. 45-50, doi: 10.1109/IWSSIP48289.2020.9145157.

[2]() D. F. D. dos Santos, A. B. Silva, P. R. de Faria, B. A. N. Travençolo and M. Z. do Nascimento, "Impacts of Color Space Transformations on Dysplastic Nuclei Segmentation Using CNN," Proceedings of the XVI Workshop de Visão Computacional, Uberlândia, Brazil, 2020, pp. 6-11.
-->