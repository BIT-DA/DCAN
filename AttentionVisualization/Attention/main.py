# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import argparse
from skimage import io
import cv2

import backbone as BackboneNetwork
from network import ImageClassifier
from interpretability.grad_cam import GradCAM
import pre_process as prep

Office31_class_name_list = []  # please fill by yourself according to your need
ImageClef_class_name_list =[]  # please fill by yourself according to your need
OfficeHome_class_name_list = ['Alarm_Clock',
                       'Backpack',
                       'Batteries',
                       'Bed',
                       'Bike',
                       'Bottle',
                       'Bucket',
                       'Calculator',
                       'Calendar',
                       'Candles',
                       'Chair',
                       'Clipboards',
                       'Computer',
                       'Couch',
                       'Curtains',
                       'Desk_Lamp',
                       'Drill',
                       'Eraser',
                       'Exit_Sign',
                       'Fan',
                       'File_Cabinet',
                       'Flipflops',
                       'Flowers',
                       'Folder',
                       'Fork',
                       'Glasses',
                       'Hammer',
                       'Helmet',
                       'Kettle',
                       'Keyboard',
                       'Knives',
                       'Lamp_Shade',
                       'Laptop',
                       'Marker',
                       'Monitor',
                       'Mop',
                       'Mouse',
                       'Mug',
                       'Notebook',
                       'Oven',
                       'Pan',
                       'Paper_Clip',
                       'Pen',
                       'Pencil',
                       'Postit_Notes',
                       'Printer',
                       'Push_Pin',
                       'Radio',
                       'Refrigerator',
                       'Ruler',
                       'Scissors',
                       'Screwdriver',
                       'Shelf',
                       'Sink',
                       'Sneakers',
                       'Soda',
                       'Speaker',
                       'Spoon',
                       'TV',
                       'Table',
                       'Telephone',
                       'ToothBrush',
                       'Toys',
                       'Trash_Can',
                       'Webcam']
Visda_class_name_list = []     # please fill by yourself according to your need

def save_original_image(image, input_image_name, output_dir):
    prefix = os.path.splitext(input_image_name)[0]
    io.imsave(os.path.join(output_dir, '{}-image.jpg'.format(prefix)), image)

# save the attention map
def save_image(image_dicts, input_image_name, network, output_dir):
    prefix = os.path.splitext(input_image_name)[0]
    for key, image in image_dicts.items():
        io.imsave(os.path.join(output_dir, '{}-{}-{}.jpg'.format(prefix, network, key)), image)

# preprocess the input
def prepare_input(image_path):
    resize_size, crop_size = 256, 224
    start_center = int((resize_size - crop_size - 1) / 2)
    img = Image.open(image_path, "r")

    input_transform1 = transforms.Compose(
        [prep.ResizeImage(resize_size), prep.PlaceCrop(crop_size, start_center, start_center)])
    image = input_transform1(img)

    input_transform2 = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    img_tensor = input_transform2(image)  # the shape of img_tensor: CxWxH

    img_tensor = img_tensor.unsqueeze(0)
    img_tensor.requires_grad_(True)
    return img_tensor, image              # the shape of img_tensor :1xCxWxH

# get the name of the last convolutional layer
def get_last_conv_name(net_name):
    if net_name == 'source_only':
        layer_name = 'backbone.layer4.2'
    else:
        raise ValueError('invalid network name:{}'.format(net_name))
    return layer_name

# normalize image
def norm_image(image):
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)

# generate cam and heatmap
def gen_cam(image, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # gbr to rgb

    # add heatmap to original image
    cam = heatmap * 0.5 + np.float32(image)
    return norm_image(cam), heatmap


def call_func(class_name_list, output_dir, net, network_name, inputs, img, image_path, true_label, want_correct):
    class_id = None
    print("network = {}".format(network_name))
    layer_name = get_last_conv_name(network_name)
    grad_cam = GradCAM(net, layer_name, class_name_list)
    mask, pseudo_label, predict_class_name, score = grad_cam(inputs, class_id)
    image_dict = {}
    if want_correct:    # get the attention map of correctly classified sample
        if pseudo_label == true_label:
            image_dict['cam'], _ = gen_cam(img, mask)
            grad_cam.remove_handlers()
            file_name = network_name + "_" + predict_class_name + "_" + "{:.3f}".format(score * 100.0)
            save_image(image_dict, os.path.basename(image_path), file_name, output_dir)
            return 1
        else:
            return 0
    else:               # get the attention map of misclassified sample
        if pseudo_label != true_label:
            image_dict['cam'], _ = gen_cam(img, mask)  # 生成CAM图和heatmap
            grad_cam.remove_handlers()
            file_name = network_name + "_" + predict_class_name + "_" + "{:.3f}".format(score * 100.0)
            save_image(image_dict, os.path.basename(image_path), file_name, output_dir)
            return 1
        else:
            return 0

def main(args):
    if args.dset == "office":
        num_classes = 31
        class_name_list = Office31_class_name_list
    elif args.dset == "image-clef":
        num_classes = 12
        class_name_list = ImageClef_class_name_list
    elif args.dset == "office-home":
        num_classes = 65
        class_name_list = OfficeHome_class_name_list
    elif args.dset == "visda":
        num_classes = 12
        class_name_list = Visda_class_name_list

    # List the number of images in each category
    domain = args.image_path.split('/')[-1]
    if args.dset == "office-home":
        if domain == 'RealWorld':
            image_num_list = [86, 99, 64, 83, 99, 78, 80, 73, 68, 99, 96, 65, 64, 76, 73, 62, 51, 43, 81, 60, 58, 85, 75, 57, 36,
                              60, 52, 60, 72, 75, 83, 78, 67, 23, 71, 46, 60, 58, 68, 64, 30, 68, 65, 59, 67, 52, 53, 66, 75, 41,
                              77, 51, 66, 77, 88, 63, 81, 54, 53, 59, 82, 85, 67, 81, 49]
        elif domain == 'Product':
            image_num_list = [79, 99, 62, 43, 44, 62, 47, 81, 81, 56, 99, 65, 96, 88, 75, 83, 67, 41, 67, 58, 71, 99, 91, 90, 41,
                              67, 57, 90, 72, 99, 41, 54, 99, 56, 98, 72, 96, 41, 93, 68, 70, 47, 60, 40, 38, 99, 43, 43, 59, 58,
                              99, 40, 49, 46, 99, 43, 99, 47, 76, 60, 58, 42, 45, 93, 98]
        elif domain == 'Art':
            image_num_list = [74, 41, 27, 40, 75, 99, 40, 33, 20, 76, 69, 25, 44, 40, 40, 23, 15, 18, 21, 45, 22, 46, 90, 20, 46,
                              40, 40, 79, 46, 18, 72, 49, 51, 20, 42, 32, 18, 49, 21, 20, 19, 19, 20, 26, 19, 18, 24, 47, 49, 15,
                              20, 30, 42, 41, 46, 40, 20, 46, 40 ,16, 44, 43, 20, 21, 16]
        else:
            print("Please add the image_num_list in the code by yourself")
    else:
        print("Please add the image_num_list in the code by yourself")

    # load pretrained model
    network_name = "source_only"
    checkpoint = torch.load(args.weight_path)
    backbone = BackboneNetwork.__dict__[args.arch](pretrained=True)
    net_source_only = ImageClassifier(backbone, num_classes).cuda()
    net_source_only.load_state_dict(checkpoint)

    for i in range(0, num_classes):
        dir = class_name_list[i]
        output_path = "./" + args.output_dir + '/' + dir
        dir_path = os.path.join(args.image_path, dir)
        if not os.path.exists(output_path):
            os.system("mkdir -p " + output_path)
        for file_order in range(1, image_num_list[i] + 1):
            file_name = "000" + str("{:02d}".format(file_order)) +".jpg"   # the format of image filename
            file_path = os.path.join(dir_path, file_name)
            print(file_path)

            # input
            inputs, img = prepare_input(file_path)  # preprocess the input images
            inputs= inputs.cuda()
            img = np.asarray(img)
            img = np.float32(img) / 255

            # save the original image
            save_original_image(img, os.path.basename(file_path), output_path)

            # get attention map
            call_func(class_name_list, output_path, net_source_only, network_name, inputs, img, file_path, i, want_correct=args.want_correct)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Attention Map')
    parser.add_argument('--arch', type=str, default='resnet50', choices=['resnet50', 'resnet101'])
    parser.add_argument('--gpu_id', type=str, nargs='?', default='3', help="device id to run")
    parser.add_argument('--dset', type=str, default='office-home', choices=['office', 'image-clef', 'visda', 'office-home'], help="The dataset dataset used")
    parser.add_argument('--image_path', type=str, default='/data1/TL/data/office-home-65/RealWorld', help="the path of raw images")
    parser.add_argument('--weight_path', type=str, default='/home/lishuang/xmx/2021CVPR_try/AnalysisExperiments/Attention/log/SourceOnly/home/Art-RealWorld_best_model.pth.tar', help="the model weight path")
    parser.add_argument('--output_dir', type=str, default='Results/RealWorld', help="the directory to save attention maps")
    parser.add_argument('--want_correct', default=True, action='store_false', help="get the attention map of correctly classified sample")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    main(args)