import cv2
import os
import argparse

# Read the video from specified path
# dir = '/home/ugh/AlphaPose/200305_gait/'
# list = os.listdir(dir)
# for d in list:
#     cam = cv2.VideoCapture(dir+d)
#     try:
#
#         # creating a folder named data
#         if not os.path.exists('frame_data200305/'+d[:-4]):
#             os.makedirs('frame_data200305/'+d[:-4])
#
#         # if not created then raise error
#     except OSError:
#         print('Error: Creating directory of data')
#
#     # frame
#     currentframe = 0
#
#     while (True):
#
#         # reading from frame
#         ret, frame = cam.read()
#
#         if ret:
#             # if video is still left continue creating images
#             name = './frame_data200305/'+d[:-4]+'/'+ str(currentframe) + '.jpg'
#             print('Creating...' + name)
#
#             # writing the extracted images
#             cv2.imwrite(name, frame)
#
#             # increasing counter so that it will
#             # show how many frames are created
#             currentframe += 1
#         else:
#             break
#
# # Release all space and windows once done
# cam.release()
# cv2.destroyAllWindows()


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

    # Release all space and windows once done
    video.release()
    video2.release()
    #cv2.destroyAllWindows()

    return
