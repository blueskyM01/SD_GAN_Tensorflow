import cv2
import numpy as np
import os
from glob import glob

def m4_image_save_cv(images, savepath, rows=4, zero_mean=True):
    # introduction: a series of images save as a picture
    # image: 4 dims
    # rows: how many images in a row
    # cols: how many images in a col
    # zero_mean:

    if zero_mean:
        images = images * 127.5 + 127.5
    if images.dtype != np.uint8:
        images = images.astype(np.uint8)
    img_num, img_height, img_width, nc = images.shape
    h_nums = rows
    w_nums = img_num // h_nums
    merge_image_height = h_nums * img_height
    merge_image_width = w_nums * img_width
    merge_image = np.ones([merge_image_height, merge_image_width, nc], dtype=np.uint8)
    for i in range(h_nums):
        for j in range(w_nums):
            merge_image[i * img_height:(i + 1) * img_height, j * img_width:(j + 1) * img_width] = images[
                i * w_nums + j]

    merge_image = cv2.cvtColor(merge_image, cv2.COLOR_BGR2RGB)  # cv2默认为bgr顺序
    cv2.imwrite(savepath, merge_image)

def m4_get_open_image_name(file_list,dataset_dir):
    for i in range(len(file_list)):
        file_list[i] = os.path.join(dataset_dir,file_list[i])
    return file_list

def m4_face_label_maker(filepath,savefilename):
    namelist = os.listdir(filepath)
    filename = filepath + '/'
    labelall = []
    idx = 0
    for name in namelist:
        imagename = []
        foldername = filename + name
        imagename = imagename + glob(foldername + '/*.jpg') + glob(foldername + '/*.png') \
               + glob(foldername + '/*.jpeg') + glob(foldername + '/*.bmp')

        for i in range(len(imagename)):
            label = [name,imagename[i].split('/')[-1],idx]
            labelall.append(label)
        idx +=1
    f = open(savefilename,'w+')
    for j in range(len(labelall)):
        f.writelines('\n'+str(labelall[j][0])+'/'+str(labelall[j][1])+ '    ' + str(labelall[j][2]))
    f.close()

def m4_get_image_name(filepath):
    namelist = os.listdir(filepath)
    filename = filepath + '/'
    imgs = []
    for name in namelist:
        name = filename + name
        imgs = imgs + glob(name + '/*.jpg') + glob(name + '/*.png')+ glob(name + '/*.jpeg')+ glob(name + '/*.bmp')
    return imgs

def m4_get_file_label_name(filepath_name,save_data_path_name):
    data = np.loadtxt(filepath_name,dtype=str)
    filename = data[:,0].tolist()
    label=data[:,1].tolist()
    filename_list = []
    label_list=[]
    for i in range(data.shape[0]):
        filename_list.append(os.path.join(save_data_path_name,filename[i].lstrip("b'").rstrip("'")))
        label_list.append(int(label[i].lstrip("b'").rstrip("'")))
    return filename_list,label_list



