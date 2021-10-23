import os
import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
import scipy
from skimage.transform import resize
from PIL import Image
import cv2
err = []
err2 = []
dir = '/home/ugh/AlphaPose/GaitDatasetB-silh/'
list = os.listdir(dir)
for d in list:
    dir2 = dir + '/' + d
    list2 = os.listdir(dir2)
    for d2 in list2:
        dir3 = dir2 + '/' + d2
        list3 = os.listdir(dir3)
        for d3 in list3:
            filename = dir3 + '/' + d3

            geiname = filename[38:]
            geiname = geiname.replace('/', '_')
            str = './GEI/_' + geiname + '.png'

            if os.path.isfile(str):
                print(str +'이미있음')
                continue

            files = os.listdir(filename)
            # images = [imread('/home/ugh/AlphaPose/GaitDatasetB-silh/001/bg-01/162/' + f) for f in files]
            images = []
            for f in files:
                img = imread(filename + '/' + f)
                images.append(img)
                print(f)
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
                    x_s = x_c - t // 2
                    x_e = x_c + t // 2
                    if (x_s < 0 or x_e > 320):
                        return None, False
                    img = img[y_s:y_e, x_s if x_s > 0 else 0:x_e if x_e < img.shape[1] else img.shape[1]]
                    # img = img[y_s:y_e, x_s:x_e]
                    return resize(img, newsize), True
                except ValueError:
                    err.append(filename)
                    return None, False


            images2 = []
            for i in images:
                a, b = image_extract(i, (128, 96))
                if b == False:
                    continue
                images2.append(a)

            try:
                gei = np.mean(images2, axis=0)
                plt.imsave(str, gei)
            except ValueError:
                err2.append(filename)



print(err)
print(err2)
