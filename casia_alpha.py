import os

command = ''

path_dir1 = '/home/ugh/AlphaPose/CASIA-B/DatasetB-1/video'
path_dir2 = '/home/ugh/AlphaPose/CASIA-B/DatasetB-2/video'

file_list1 = os.listdir(path_dir1)
file_list2 = os.listdir(path_dir2)

file_list1.sort()
file_list2.sort()

for item in file_list1:
    command = './scripts/inference.sh ./configs/coco/hrnet/256x192_w32_lr1e-3.yaml ' \
              './pretrained_models/hrnet_w32_256x192.pth ./CASIA-B/DatasetB-1/video/' + item + ' ./CASIA_B_result/' + item[:-4]
    os.mkdir('./CASIA_B_result/' + item[:-4] + '/')
    f = open('./CASIA_B_result/' + item[:-4] + '/alphapose-results.json', 'w')
    f.close()
    os.system(command)

for item in file_list2:
    command = './scripts/inference.sh ./configs/coco/hrnet/256x192_w32_lr1e-3.yaml ' \
              './pretrained_models/hrnet_w32_256x192.pth ./CASIA-B/DatasetB-2/video/' + item + ' ./CASIA_B_result/' + item[:-4]
    os.mkdir('./CASIA_B_result/' + item[:-4] + '/')
    f = open('./CASIA_B_result/' + item[:-4] + '/alphapose-results.json', 'w')
    f.close()
    os.system(command)


