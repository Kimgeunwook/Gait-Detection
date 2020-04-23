import cv2
import os

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
cam = cv2.VideoCapture('/home/ugh/AlphaPose/200302_gait/jiho_090.MOV')
currentframe = 0
while (True):

        # reading from frame
        ret, frame = cam.read()
        if ret:
            # if video is still left continue creating images
            name = './oclussion_frame/'+str(currentframe)+'.jpg'
            print('Creating...' + str(currentframe)+'.jpg')

            # writing the extracted images
            cv2.imwrite(name, frame)
            currentframe += 1

        else:
            break

# Release all space and windows once done
cam.release()
cv2.destroyAllWindows()