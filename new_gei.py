import os
import argparse
import cv2
import video_to_frame
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

##################### Mask RCNN Initializing ###########################################################################
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Root directory of the project
ROOT_DIR = os.path.abspath("Mask_RCNN/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
#IMAGE_DIR = os.path.join(ROOT_DIR, "ugh")


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
# class_names = ['BG', 'person']


def mask_rcnn(directory):
    # Load a random image from the images folder
    roi = []
    cycle = []
    file_names = os.listdir(directory)
    file_names.sort()
    for file in file_names:
        if file[-7:-4] == 'img':
            image = skimage.io.imread(os.path.join(directory, file))
            (height, width, dim) = image.shape

            # Run detection
            results = model.detect([image], verbose=1)

            # Visualize results
            r = results[0]
            # person_list = []
            for i in range(len(r['rois'])):
                if r['class_ids'][i] == 1:
                    aaa = r['rois'][i]
                    roi.append(aaa)
                    if(aaa[0] > 10 and aaa[1] > 10 and aaa[2] < height-10 and aaa[3] < width - 10):
                        cycle.append(file)
                    # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                    #                             class_names, r['scores'])

            # from matplotlib import pyplot as plt
            # for i in range(len(r['class_ids'])):
            #     if r['class_ids'][i] == 1:
            #         arr = np.array(r['masks'][:,:,i]).astype('float') * 255
            #         arr=arr.astype('uint8')
            #         print(np.shape(arr))
            #         plt.imsave('/home/ugh/AlphaPose/vvvvvv.png', arr)
            #         print(arr)

    return roi, cycle
#####################################################################################################################

parser = argparse.ArgumentParser(description='Background Matting.')
parser.add_argument('-v', '--video_path', type=str, required=True, help='File path of the input video (required)')
parser.add_argument('-m', '--trained_model', type=str, default='real-fixed-cam',choices=['real-fixed-cam', 'real-hand-held', 'syn-comp-adobe'],help='Trained background matting model')
parser.add_argument('-o', '--output_dir', type=str, required=True,help='Directory to save the output results. (required)')
#parser.add_argument('-i', '--input_dir', type=str, required=True,help='Directory to load input images. (required)')
parser.add_argument('-tb', '--target_back', type=str,help='Directory to load the target background.')

args = parser.parse_args()

v = args.video_path
# i = args.input_dir
o = args.output_dir
tb = args.target_back
m = args.trained_model


# os.system('python video_to_frame.py -i ' + v)
video_to_frame.frame(v)

rois, gait = mask_rcnn('./frames/input')
print(rois)
print(gait)


#TODO# mask_rcnn 돌려서 roi, gait cycle 잡아둠
#
#
# os.system('python test_segmentation_deeplab.py -i ./frames/input')
#
# #os.system('python test_pre_process.py -i test/input')
# os.system('python test_background-matting_image.py -i ./frames/input -o ' + o + ' -tb test/background/0001.png')
#
# os.system('rm -r -f ./frames')

