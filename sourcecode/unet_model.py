import torch
import torch.nn as nn
import numpy as np

from sourcecode.logger_utils import *


class UNetBaseConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(UNetBaseConv, self).__init__()

        self.unet_act = nn.ReLU(inplace=True)
        #self.unet_act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.unet_norm = nn.BatchNorm2d(out_channels)

        self.unet_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        nn.init.normal_(self.unet_conv1.weight, mean=0.0, std=np.sqrt(2/(kernel_size*kernel_size*in_channels)))

        self.unet_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        nn.init.normal_(self.unet_conv2.weight, mean=0.0, std=np.sqrt(2/(kernel_size*kernel_size*in_channels)))

    def forward(self, x):

        x = self.unet_conv1(x)
        x = self.unet_norm(x)
        x = self.unet_act(x)
        
        x = self.unet_conv2(x)
        x = self.unet_norm(x)
        x = self.unet_act(x)

        return x


class UNetBaseConvUp(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UNetBaseConvUp, self).__init__()

        self.unet_upsample = nn.Upsample(mode='bilinear', scale_factor=2, align_corners=True)
        self.unet_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.unet_upsample(x)
        x = self.unet_conv2d(x)
        return x


class UNetDownConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(UNetDownConv, self).__init__()

        self.unet_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.unet_conv_block = UNetBaseConv(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.unet_pool(x)
        x = self.unet_conv_block(x)
        return x


class UNetUpConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, up_mode):
        super(UNetUpConv, self).__init__()
        assert up_mode in ('ConvTranspose', 'Upsample')

        self.up_mode = up_mode
        self.unet_up_conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        #self.unet_up_conv_upsample = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2, align_corners=True), nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.unet_up_conv_upsample = UNetBaseConvUp(in_channels=in_channels, out_channels=out_channels)
        self.unet_conv_block = UNetBaseConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x, crop):
        x = self.unet_up_conv_transpose(x) if self.up_mode == 'ConvTranspose' else self.unet_up_conv_upsample(x)
        x = torch.cat([x, crop], dim=1)
        x = self.unet_conv_block(x)
        return x


class UNetClassifier(nn.Module):
    def __init__(self, in_features, out_features):
        super(UNetClassifier, self).__init__()

        self.unet_linear_01 = nn.Linear(in_features, 512)
        self.unet_linear_02 = nn.Linear(512, 512)
        self.unet_linear_03 = nn.Linear(512, out_features)
        
        self.unet_act_tanh = nn.Tanh()
        self.unet_act_sigmoid = nn.Sigmoid()
        self.unet_act_relu = nn.ReLU(inplace=True)
        self.unet_act_softmax = nn.Softmax()

    def forward(self, x):

        #print("classifier -> x.shape: {}".format(x.shape))
        x = self.unet_linear_01(x)
        x = nn.Dropout()(x)
        x = self.unet_act_relu(x)

        x = self.unet_linear_02(x)
        x = nn.Dropout()(x)
        x = self.unet_act_relu(x)

        x = self.unet_linear_03(x)
        x = nn.Dropout()(x)
        x = self.unet_act_softmax(x)

        return x


def center_crop(layer, target_size_x, target_size_y):
    lower_x = int((layer.shape[2] - target_size_x) / 2)
    upper_x = lower_x + target_size_x
    lower_y = int((layer.shape[3] - target_size_y) / 2)
    upper_y = lower_y + target_size_y

    return layer[:, :, lower_x:upper_x, lower_y:upper_y]


class UNet(nn.Module):
    def __init__(self, in_channels=1,
                 out_channels=2,
                 up_mode='ConvTranspose',
                 padding=False,
                 img_input_size=(160, 160),
                 out_features=-1):
        """
        Implementation of U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015)
        https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/

        -- Args
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        up_mode (str): one of 'ConvTranspose' or 'Upsample':
                           - 'ConvTranspose' will use transposed convolutions for learned upsampling.
                           - 'Upsample' will use bilinear upsampling.
        padding:
        img_input_size:
        out_features:
        """
        super(UNet, self).__init__()
        assert up_mode in ('ConvTranspose', 'Upsample')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_mode = up_mode
        self.pad = 1 if padding else 0
        self.img_input_size = img_input_size
        self.img_input_size_decode = (int(img_input_size[0]/16), int(img_input_size[1]/16))
        self.out_features = out_features
        
        self.init_conv = UNetBaseConv(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=self.pad)

        self.encode1 = UNetDownConv(64, 64*2, kernel_size=3, stride=1, padding=self.pad)
        self.encode2 = UNetDownConv(64*2, 64*2*2, kernel_size=3, stride=1, padding=self.pad)
        self.encode3 = UNetDownConv(64*2*2, 64*2*2*2, kernel_size=3, stride=1, padding=self.pad)
        self.encode4 = UNetDownConv(64*2*2*2, 64*2*2*2*2, kernel_size=3, stride=1, padding=self.pad)

        self.decode1 = UNetUpConv(64*2*2*2*2, 64*2*2*2, kernel_size=3, stride=1, padding=self.pad, up_mode=up_mode)
        self.decode2 = UNetUpConv(64*2*2*2, 64*2*2, kernel_size=3, stride=1, padding=self.pad, up_mode=up_mode)
        self.decode3 = UNetUpConv(64*2*2, 64*2, kernel_size=3, stride=1, padding=self.pad,  up_mode=up_mode)
        self.decode4 = UNetUpConv(64*2, 64, kernel_size=3, stride=1, padding=self.pad, up_mode=up_mode)

        self.exit_conv = nn.Conv2d(64, out_channels, kernel_size=1, stride=1, padding=0)

        if self.out_features > 0:
            # the image in the last encode layer: img_input_size x img_input_size and 1 channel
            self.classifier = UNetClassifier(in_features=self.img_input_size[0]*self.img_input_size[1]*1, out_features=out_features)

    def forward(self, x):

        if self.img_input_size[0] != x.shape[-2] or self.img_input_size[1] != x.shape[-1]:
            logger.warn("!!!! Input shape does not match! {} / {} !!!!".format(x.shape, self.img_input_size))

        # encode (down)
        x = self.init_conv(x)
        row_1_aux = x

        x = self.encode1(x)
        row_2_aux = x

        x = self.encode2(x)
        row_3_aux = x

        x = self.encode3(x)
        row_4_aux = x

        x = self.encode4(x)

        x = nn.Dropout()(x)

        # decode (up)
        # print("x.shape: {}".format(x.shape))
        # print("row_4_aux.shape: {}".format(row_4_aux.shape))
        crop = center_crop(row_4_aux, int(x.shape[2] * 2), int(x.shape[3] * 2))
        # print("crop.shape: {}".format(crop.shape))
        x = self.decode1(x, crop)

        crop = center_crop(row_3_aux, int(x.shape[2] * 2), int(x.shape[3] * 2))
        x = self.decode2(x, crop)

        crop = center_crop(row_2_aux, int(x.shape[2] * 2), int(x.shape[3] * 2))
        x = self.decode3(x, crop)

        crop = center_crop(row_1_aux, int(x.shape[2] * 2), int(x.shape[3] * 2))
        x = self.decode4(x, crop)

        x = self.exit_conv(x)

        x = nn.Sigmoid()(x)

        if self.out_features > 0:
            
            #print("x.shape: {}".format(x.shape))
            x = x.view(x.size(0), -1)
            x = self.classifier(x)

        return x;

    def name(self):
        return "UNet_IN-{}_OUT-{}_UPMODE-{}".format(self.in_channels, self.out_channels, self.up_mode)
    
    def model_input_channels(self):
        return self.in_channels
    
    def model_output_channels(self):
        return self.out_channels
    
    def model_up_mode(self):
        return self.up_mode
    
    def model_padding(self):
        return self.pad


def load_checkpoint(file_path='checkpoints/UNet_IN-1_OUT-2_UPMODE-ConvTranspose_Epoch-1000.pt',
                    img_input_size=(640, 640),
                    use_cuda=True):

    # Checking for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if use_cuda else "cpu"

    # load the trained model
    checkpoint = torch.load(file_path) if torch.cuda.is_available() else torch.load(file_path, map_location=lambda storage, loc: storage)

    model_in_channels = checkpoint['model_in_channels']
    model_out_channels = checkpoint['model_out_channels']
    model_up_mode = checkpoint['model_up_mode']
    model_padding = key_check('model_padding', checkpoint, True)

    # recreate the model
    with torch.no_grad():
        model = UNet(in_channels=model_in_channels,
                     out_channels=model_out_channels,
                     up_mode=model_up_mode,
                     padding=model_padding,
                     img_input_size=img_input_size).to(device) if use_cuda else UNet(in_channels=model_in_channels,
                                                                                     out_channels=model_out_channels,
                                                                                     up_mode=model_up_mode,
                                                                                     padding=model_padding,
                                                                                     img_input_size=img_input_size)
        model.load_state_dict(checkpoint['model_state_dict'])

    logger.info('\t Model loaded on: {} / {} / {} / {} / {} params -> {}'.format(device,
                                                                                 model_in_channels,
                                                                                 model_out_channels,
                                                                                 img_input_size,
                                                                                 count_parameters(model),
                                                                                 file_path))
    return model


def key_check(key, arr, default):
    if key in arr.keys():
        return arr[key]
    return default


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
