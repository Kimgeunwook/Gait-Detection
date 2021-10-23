import cv2
import os
import argparse


def frame(input_dir):

    print('start')
    bkgrd = input_dir[-17:-14] + '-bkgrd-' + input_dir[-7:]

    video = cv2.VideoCapture(input_dir)
    currentframe = 0

    if not os.path.exists('/home/ugh/AlphaPose/Background-Matting/frames/input'):
        os.makedirs('/home/ugh/AlphaPose/Background-Matting/frames/input')
    if not os.path.exists('/home/ugh/AlphaPose/Background-Matting/frames/output'):
        os.makedirs('/home/ugh/AlphaPose/Background-Matting/frames/output')
    while(True):
        # reading from frame
        ret, frame = video.read()

        if ret:
            # if video is still left continue creating images
            name = '/home/ugh/AlphaPose/Background-Matting/frames/input/'+str(currentframe)+'_img.png'
            print('Creating...' + str(currentframe)+'_img.png')

            # writing the extracted images
            cv2.imwrite(name, frame)
            currentframe += 1
        else:
            break

    bg_dir = input_dir[:-17] + bkgrd
    print(bg_dir)
    video2 = cv2.VideoCapture(bg_dir)
    count = 0
    ret, frame = video2.read()

    for i in range(currentframe):
        name = '/home/ugh/AlphaPose/Background-Matting/frames/input/'+str(count)+'_back.png'
        print('Creating...' + str(count)+'_back.png')
        cv2.imwrite(name, frame)
        count += 1
    video.release()
    video2.release()

    return
