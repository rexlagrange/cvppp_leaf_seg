import h5py
import cv2
import os
import datetime
from os.path import normpath, basename,join

def _hdf5_to_image(h5file, folder, plant_id, save_path,img_name='rgb'):
    image_arr = h5file[folder][plant_id][img_name][()]
    image_file_name = h5file[folder][plant_id][img_name+'_filename'][()].decode()
    image_path = f'{save_path}/{folder}/{image_file_name}'
    cv2.imwrite(image_path, image_arr) 
def hdf5_to_images(file_path,data='training',save_path='./test_set'):
    with h5py.File(file_path,'r') as f:
        for folder in f.keys():
            os.makedirs(save_path+'/' + folder,exist_ok=True)
            for plant_id in f[folder].keys():
                if data=='training' or data=='test':
                    _hdf5_to_image(f, folder, plant_id, save_path)
                if data == 'training':
                    _hdf5_to_image(f, folder, plant_id, save_path,'fg')
                    _hdf5_to_image(f, folder, plant_id, save_path,'centers')
                if data == 'training_truth':
                    _hdf5_to_image(f, folder, plant_id, save_path,'label')
                    #这个h5里还有个count应该是树叶的个数，后面用不太上就不导出了
                    
def images_to_hdf5(folder_paths,save_path):
    if not os.path.exists(save_path): 
        os.mkdir(save_path)
    with h5py.File(join(save_path,basename(save_path)+".h5"),'w') as f:
        for folder_path in folder_paths:
            folder_group = f.create_group(basename(normpath(folder_path)))
            for img_name in os.listdir(folder_path):
                img_path = join(folder_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                img_group_name = img_name.split('_')[0]
                img_group = folder_group.create_group(img_group_name)
                img_group.create_dataset('label', shape=img.shape, data=img)
                img_group.create_dataset('label_filename', data=f'{img_group_name}_label.png')
def batch_hdf52imgs():
    hdf5_to_images('/Volumes/Intel2T/data/cvppp/CVPPP2017_testing_images.h5','testing','/Volumes/Intel2T/data/cvppp/CVPPP2017_LSC/test')
    hdf5_to_images('/Volumes/Intel2T/data/cvppp/CVPPP2017_training_images.h5','training','/Volumes/Intel2T/data/cvppp/CVPPP2017_LSC/train')
    hdf5_to_images('/Volumes/Intel2T/data/cvppp/CVPPP2017_training_truth.h5','training_truth','/Volumes/Intel2T/data/cvppp/CVPPP2017_LSC/label')

def batch_imgs2hdf5():
    convert_dir=r"D:\predict\test"
    for yolomodel in os.listdir(convert_dir):
        folder_paths=[join(convert_dir,yolomodel,group) for group in ["A1","A2","A3","A4","A5"]]
        dest=join(r"D:\h5",yolomodel)
        images_to_hdf5(folder_paths,dest)
def imgs2hdf5_1():#单个模型
    convert_dir=r"D:\predict\test"
    yolomodel="YOLOv8l-seg-BiFPN_Ghostconv2"
    folder_paths=[join(convert_dir,yolomodel,group) for group in ["A1","A2","A3","A4","A5"]]
    dest=join(r"D:\h5",yolomodel)
    images_to_hdf5(folder_paths,dest)
if __name__ == '__main__':
    imgs2hdf5_1()   
    #batch_imgs2hdf5()