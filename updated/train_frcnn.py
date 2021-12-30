import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
import tensorflow as tf
import keras
import random
from vgg import * 
from pascal_voc_parser import *
from data_generators import *
import config as config

'''
    Big changes made:
    changed K.common.image_dim_ordering() => K.image_data_format()
        => returns 'channels_first' or 'channels_last'
            => channels_first == th
            => channels_last == tf

'''


sys.setrecursionlimit(40000)

# Parse input for training

parser = OptionParser()

parser.add_option("-p", "--path", dest="train_path", help="Path to training data.")
parser.add_option("-o", "--parser", dest="parser", help="Parser to use. One of simple or pascal_voc",
				default="pascal_voc")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois", help="Number of RoIs to process at once.", default=32)
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.", default='resnet50')
parser.add_option("--hf", dest="horizontal_flips", help="Augment with horizontal flips in training. (Default=false).", action="store_true", default=False)
parser.add_option("--vf", dest="vertical_flips", help="Augment with vertical flips in training. (Default=false).", action="store_true", default=False)
parser.add_option("--rot", "--rot_90", dest="rot_90", help="Augment with 90 degree rotations in training. (Default=false).",
				  action="store_true", default=False)
parser.add_option("--num_epochs", type="int", dest="num_epochs", help="Number of epochs.", default=2000)
parser.add_option("--config_filename", dest="config_filename", help=
				"Location to store all the metadata related to the training (to be used when testing).",
				default="config.pickle")
parser.add_option("--output_weight_path", dest="output_weight_path", help="Output path for weights.", default='./model_frcnn.hdf5')
parser.add_option("--input_weight_path", dest="input_weight_path", help="Input path for weights. If not specified, will try to load default weights provided by keras.")

(options, args) = parser.parse_args()

# Set paremeters
C = config.Config()
C.network = 'vgg'
C.num_rois = int(options.num_rois)
C.base_net_weights = get_weight_path()
C.model_path = options.output_weight_path

# Ignore input for now lol
train_dir = "../VOC2012"

# Use old data parser, unsure how to split train and test with function
train_imgs, classes_count, class_mapping = get_data(train_dir, 'trainval')
val_imgs, _, _ = get_data(train_dir, 'test')

C.class_mapping = class_mapping
inv_map = {v: k for k, v in class_mapping.items()} # not sure what this does

# Shuffle training data
random.shuffle(train_imgs)
num_imgs = len(train_imgs)

data_gen_train = get_anchor_gt(train_imgs, classes_count, C, get_img_output_length, keras.backend.image_data_format(), mode='train')
data_gen_val = get_anchor_gt(val_imgs, classes_count, C, get_img_output_length, keras.backend.image_data_format(), mode='val')

if keras.backend.image_data_format()== 'th':
	input_shape_img = (3, None, None)
else:
	input_shape_img = (None, None, 3)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(None, 4))

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = rpn(shared_layers, num_anchors)

classifier = classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count), trainable=True)

model_rpn = Model(img_input, rpn[:2])
model_classifier = Model([img_input, roi_input], classifier)

# this is a model that holds both the RPN and the classifier, used to load/save weights for the models
model_all = Model([img_input, roi_input], rpn[:2] + classifier)