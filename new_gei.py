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
import shutil
from skimage.transform import resize
##################### tensorflow Initializing ###########################################################################
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


##################### Mask RCNN Initializing ###########################################################################

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
    print("캬캬캬캬"+str(directory))
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
            #print('ㅇ아아아아아아아아아아아')
            #print(r)
            # person_list = []
            if (r['class_ids'].tolist()).count(1) != 1:
                os.remove(directory + '/' + file)
                bgfile = file[0:-7] + 'back.png'
                os.remove(directory + '/' + bgfile)
                while True:
                    print('while')
                    if not os.path.isfile(directory + '/' + bgfile):
                        break
                continue
            count_95 = 0
            for j in range(len(r['rois'])):
                if r['class_ids'][j] == 1:
                    if r['scores'][j] > 0.99:
                        aaa = r['rois'][j]
                        count_95 = count_95 + 1
                        if(aaa[0] > 10 and aaa[1] > 10 and aaa[2] < height-10 and aaa[3] < width - 10):
                            roi.append(aaa)
                            cycle.append(file)

                        else:
                            os.remove(directory + '/' + file)
                            bgfile = file[0:-7] + 'back.png'
                            os.remove(directory + '/' + bgfile)
                            while True:
                                print('while')
                                if not os.path.isfile(directory + '/' + bgfile):
                                    break

            if count_95 != 1:

                os.remove(directory + '/' + file)
                bgfile = file[0:-7] + 'back.png'
                os.remove(directory + '/' + bgfile)
                while True:
                    print('while')
                    if not os.path.isfile(directory + '/' + bgfile):
                        break

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

from test_segmentation_deeplab import segmentation, seg_init
from test_pre_process import allignment
from test_background_matting_image import matting
MODEL = seg_init()

parser = argparse.ArgumentParser(description='Background Matting.')
#parser.add_argument('-v', '--video_path', type=str, required=True, help='File path of the input video (required)')
parser.add_argument('-m', '--trained_model', type=str, default='real-fixed-cam',choices=['real-fixed-cam', 'real-hand-held', 'syn-comp-adobe'],help='Trained background matting model')
parser.add_argument('-o', '--output_dir', type=str, required=True,help='Directory to save the output results. (required)')
parser.add_argument('-i', '--input_dir', type=str, required=True,help='Directory to load input images. (required)')
parser.add_argument('-tb', '--target_back', type=str,help='Directory to load the target background.')

args = parser.parse_args()

#v = args.video_path
i = args.input_dir
o = args.output_dir
tb = args.target_back
m = args.trained_model
err = []
file_dir_path = ['/home/ugh/AlphaPose/CASIA-B/DatasetB-1/video/','/home/ugh/AlphaPose/CASIA-B/DatasetB-2/video/']

for aa in file_dir_path:
    list = os.listdir(aa)
    for v in list:
        if v[-7:-4] != '090':
            continue
        if v.count('bkgrd') != 0:
            continue
        if os.path.isfile('/home/ugh/AlphaPose/Background-Matting/GEI/'+v.split("/")[-1].replace("avi","png")):
            print(v + '이미있음')
            continue
        if not os.path.exists('/home/ugh/AlphaPose/Background-Matting/frames'):
            os.makedirs('/home/ugh/AlphaPose/Background-Matting/frames')
        print(v)
        # os.system('python video_to_frame.py -i ' + v)
        video_to_frame.frame(aa + v)

        rois, gait = mask_rcnn(i)
        print(rois)
        print(gait)
        # print(len(rois))
        # print(len(gait))


        segmentation(i, MODEL)
        #allignment(i)
        try:
            matting(i, o)
        except ValueError:
            continue
        ##########################################gei 만들
        def mass_center(img, is_round=True):
            Y = img.mean(axis=1)
            X = img.mean(axis=0)
            Y_ = np.sum(np.arange(Y.shape[0]) * Y) / np.sum(Y)
            X_ = np.sum(np.arange(X.shape[0]) * X) / np.sum(X)
            if is_round:
                return int(round(X_)), int(round(Y_))
            return X_, Y_


        def image_extract(img, newsize):
            try:
                x_s = np.where(img.mean(axis=0) != 0)[0].min()
                x_e = np.where(img.mean(axis=0) != 0)[0].max()
                y_s = np.where(img.mean(axis=1) != 0)[0].min()
                y_e = np.where(img.mean(axis=1) != 0)[0].max()
                t = int((y_e - y_s) * (newsize[1] / newsize[0]))
                x_c, _ = mass_center(img)
                #     x_c = (x_s+x_e)//2
                # x_s = x_c - t // 2)
                # x_e = x_c + t // 2)
                x_s = x_c - t // 2
                x_e = x_c + t // 2
                if (x_s < 0 or x_e > 320):
                    return None, False
                img = img[y_s:y_e, x_s if x_s > 0 else 0:x_e if x_e < img.shape[1] else img.shape[1]]
                # img = img[y_s:y_e, x_s:x_e]
                return resize(img, newsize), True
            except ValueError:
                err.append(v)
                return None, False


        gei_list = []
        for ff in range(len(gait)):
            image_path = o + '/' + gait[ff].replace("img","out")
            print(image_path)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            print(image)
            height, width = image.shape
            image = image[ rois[ff][0]:rois[ff][2] , rois[ff][1]:rois[ff][3] ]
            zero_array = np.zeros((height,width))
            zero_array[10: rois[ff][2] - rois[ff][0] + 10 ,100: rois[ff][3] - rois[ff][1] + 100] = image
            a, b = image_extract(zero_array, (128, 96))
            if b == False:
                continue
            gei_list.append(a)

        gei = np.mean(gei_list, axis=0)
        plt.imsave('/home/ugh/AlphaPose/Background-Matting/GEI/' + v.split("/")[-1].replace("avi","png") , gei)

        os.system('rm -r -f /home/ugh/AlphaPose/Background-Matting/frames/')
        #shutil.rmtree('./frames')

        while True:
            print('while')
            if not os.path.isdir('/home/ugh/AlphaPose/Background-Matting/frames'):
                break

print('done')
print(err)
