import sys
from mask2yolo import yolo2mask
import cv2
import os
from os.path import basename,join
import numpy as np
from evaluate_LSC import evaluate_without_h5_nosubfolder

from ultralytics import YOLO

def my_yolo_basline1():# 0.8519354568 
    model = YOLO(r"D:\wp\workspace\yolo8\peng_mod\cvppp\yolov8l-seg.yaml") 
    results = model.train(data=r"D:\wp\workspace\yolo8\peng_mod\cvppp\coco128-seg.yaml",epochs=250, imgsz=512,batch=16,workers=4,single_cls=True,degrees=180,flipud=0.5)
def my_yolo_basline2(): #0.8550422167
    model = YOLO(r"D:\wp\workspace\yolo8\peng_mod\cvppp\yolov8l-seg-BiFPN.yaml") 
    results = model.train(data=r"D:\wp\workspace\yolo8\peng_mod\cvppp\coco128-seg.yaml",epochs=250, imgsz=512,batch=16,workers=4,single_cls=True,degrees=180,flipud=0.5)
def my_yolo_basline3(): #0.8636059322
    model = YOLO(r"D:\wp\workspace\yolo8\peng_mod\cvppp\yolov8l-seg-Ghostconv.yaml") 
    results = model.train(data=r"D:\wp\workspace\yolo8\peng_mod\cvppp\coco128-seg.yaml",epochs=250, imgsz=512,batch=16,workers=4,single_cls=True,degrees=180,flipud=0.5)

def _special_add(np1,np2,no_overlap=True):
    #cumu=0
    if no_overlap:
        np1m=np1.astype(bool)
        np2m=np2.astype(bool)
        overlap=np1m&np2m
        #cumu+=sum(overlap.flatten())
        np2[overlap]=0
        np3=np1+np2
    else:
        np3=np1+np2
    #print(cumu)
    return np3.astype(np.uint8)



def predict_mask(model_path,source,dest):
    if not os.path.exists(dest):
        os.makedirs(dest)
    model = YOLO(model_path)
    #source支持目录和文件pil图片等
    results=model(source=source,save=True,show_labels=False,show_conf=False,boxes=False,retina_masks=True)
    for result in results:
        image_name = basename(result.path)  # 提取图片名称
        mask_name = image_name.replace("_rgb.png",'_label.png')  # 根据图片名称生成保存结果的名称
        pred_image_path = join(dest, mask_name)# 图片保存路径
        print(pred_image_path)
        w,h=result.orig_img.shape[1],result.orig_img.shape[0]
        mask_overlap=np.zeros((h,w),dtype=np.uint8) #mask_overlap是和result.orig_img一样大小的单通道全0矩阵
        #mask_overlap=cv2.cvtColor(result.orig_img, cv2.COLOR_RGB2GRAY)#debug
        #如果有树叶
        if result.masks is not None and len(result.masks) > 0:
            masks_data = result.masks.data
            i=1
            for index, mask in enumerate(masks_data):#可能是多个树叶
                mask = mask.cpu().numpy()
                #mask的大小转换成和原图一样
                mask = cv2.resize(mask, (w, h))
                # 保持mask*i的值是uint8
                mask=(mask*i).astype(np.uint8)
                mask_overlap=_special_add(mask_overlap,mask)
                i+=1
            cv2.imwrite(pred_image_path , mask_overlap)

def predict_test2017():#预测cvppp训练集中分出来的1/4测试集
    result_str=""
    model_dir=r"D:\wp\workspace\weights"
    for weight in os.listdir(model_dir):
        model_path=join(model_dir,weight,"weights","best.pt")
        dest=join(r"D:\predict\test2017",weight)
        source=r"D:\wp\workspace\coco_cvppp_31\images\test2017"
        predict_mask(model_path,source,dest)
        results, stats=evaluate_without_h5_nosubfolder(dest,r"D:\wp\workspace\coco_cvppp_31\labels\png\test2017")
        print(weight)
        print(stats)
        result_str+=(weight+'\n'+str(stats)+'\n')
    
    print(result_str)
    with open(r"D:\predict\test2017\result.txt",'w') as f:
        f.write(result_str)
def predict_test2017_1():#预测cvppp训练集中分出来的1/4测试集
    result_str=""
    model_dir=r"D:\wp\workspace\weights"
    weight ="YOLOv8l-seg-BiFPN_Ghostconv"
    model_path=join(model_dir,weight,"weights","best.pt")
    dest=join(r"D:\predict\test2017",weight)
    source=r"D:\wp\workspace\coco_cvppp_31\images\test2017"
    predict_mask(model_path,source,dest)
    results, stats=evaluate_without_h5_nosubfolder(dest,r"D:\wp\workspace\coco_cvppp_31\labels\png\test2017")
    print(weight)
    print(stats)
    result_str+=(weight+'\n'+str(stats)+'\n')
    
    print(result_str)
    with open(r"D:\predict\test2017\result2.txt",'w') as f:
        f.write(result_str)

def predict_testset():#预测cvppp原来的测试集
    model_dir=r"D:\wp\workspace\weights"
    for weight in os.listdir(model_dir):
        model_path=join(model_dir,weight,"weights","best.pt")
        dest=join(r"D:\predict\test",weight)
        source=r"D:\wp\workspace\coco_cvppp\test"
        for group in ["A1","A2","A3","A4","A5"]:
            predict_mask(model_path,join(source,group),join(dest,group))
def predict_1():#用指定的模型预测测试集
    model_dir=r"D:\wp\workspace\weights"
    weight='YOLOv8l-seg_aug'
    model_path=join(model_dir,weight,"weights","best.pt")
    dest=join(r"D:\predict\test",weight)
    source=r"D:\wp\workspace\coco_cvppp\test"
    for group in ["A1","A2","A3","A4","A5"]:
        predict_mask(model_path,join(source,group),join(dest,group))

    
if __name__ == '__main__':
    my_yolo_basline1()
    #my_yolo_basline2()
    #my_yolo_basline3()
    #predict_1()

