import numpy as np
import cv2
from PIL import Image
from os.path import normpath, basename,join
import os
# 把预测的mask转换成彩色的mask
def colorful(mask,save_path):
    mask = Image.fromarray(mask) 
    palette=[]
    for i in range(256):
        palette.extend((i,i,i))
    palette[:3*21]=np.array([   [0, 0, 0],
                                [128, 0, 0],
                                [0, 128, 0],
                                [128, 128, 0],
                                [0, 0, 128],
                                [128, 0, 128],
                                [0, 128, 128],
                                [128, 128, 128],
                                [64, 0, 0],
                                [192, 0, 0],
                                [64, 128, 0],
                                [192, 128, 0],
                                [64, 0, 128],
                                [192, 0, 128],
                                [64, 128, 128],
                                [192, 128, 128],
                                [0, 64, 0],
                                [128, 64, 0],
                                [0, 192, 0],
                                [128, 192, 0],
                                [0, 64, 128]
                             ], dtype='uint8').flatten()
 
    mask.putpalette(palette)
    mask.save(save_path)
def batch_colorful():
    convert_dir=r"D:\predict\test\yolov8l-seg_aug"
    save_dir=os.path.join(r"D:\label_color",basename(convert_dir))
    if not os.path.exists(save_dir): 
        os.mkdir(save_dir)
    folder_paths=[join(convert_dir,group) for group in ["A1","A2","A3","A4"]]
    for folder in folder_paths:
        if not os.path.exists(join(save_dir,basename(folder))): 
            os.mkdir(join(save_dir,basename(folder)))
        for mask in os.listdir(folder):
            if mask.endswith('.png') and not mask.startswith('._'):
                print(os.path.join(folder,mask))
                img = cv2.imread(os.path.join(folder,mask),cv2.IMREAD_GRAYSCALE)
                colorful(img, os.path.join(save_dir,basename(folder),mask))
if __name__ == '__main__':
    batch_colorful()
    