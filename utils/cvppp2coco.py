import os
# 1 convertH5.main() 把H5转换成png文件
# 2 mask2yolo.main() 把label目录中的mask的png转换成yolo的txt格式存储在相同目录下
# 3 cvppp2coco.main() 把原目录下的png和txt按coco的目录结构复制到新的目录下


def alltrain():
    dest_dir='/Volumes/Intel2T/data/cvppp/CVPPP2017_LSC/coco_cvppp_all/'
    dest_train_images=dest_dir + 'images/train2017/'
    dest_train_labels=dest_dir + 'labels/train2017/'
    train_sources='/Volumes/Intel2T/data/cvppp/CVPPP2017_LSC/train/'
    label_sources='/Volumes/Intel2T/data/cvppp/CVPPP2017_LSC/label/'
    groups=['A1','A2','A3','A4']
    for group in groups:
        for img in os.listdir(train_sources + group):
            if img.endswith('rgb.png'):
                str=f'ditto -V {train_sources + group + "/" + img} {dest_train_images + group + img}'
                # ditto 只在mac上有，别的系统可以用cp 但是cp有个缺点需要自己先把目标路径的目录手动创建好，因为cp不会自动创建目录
                str2=str.replace(train_sources, label_sources).replace(dest_train_images,dest_train_labels).replace('rgb.png','rgb.txt')
                os.system(str)
                os.system(str2)
 
def quater_test():
    # 1/4的数据集作为测试集（验证集）
    dest_dir='/Volumes/Intel2T/data/cvppp/CVPPP2017_LSC/coco_cvppp_31/'
    dest_train_images=dest_dir + 'images/train2017/'
    dest_train_labels=dest_dir + 'labels/train2017/'
    dest_test_images=dest_dir + 'images/test2017/'
    dest_test_labels=dest_dir + 'labels/test2017/'
    train_sources='/Volumes/Intel2T/data/cvppp/CVPPP2017_LSC/train/'
    label_sources='/Volumes/Intel2T/data/cvppp/CVPPP2017_LSC/label/'
    groups=['A1','A2','A3','A4']
    tmp=0
    for group in groups:
        for img in os.listdir(train_sources + group):
            if img.endswith('rgb.png'):
                if tmp%4==0:
                    str=f'ditto -V {train_sources + group + "/" + img} {dest_test_images + group + img}'
                    str2=str.replace(train_sources, label_sources).replace(dest_test_images,dest_test_labels).replace('rgb.png','rgb.txt')
                else:
                    str=f'ditto -V {train_sources + group + "/" + img} {dest_train_images + group + img}'
                    str2=str.replace(train_sources, label_sources).replace(dest_train_images,dest_train_labels).replace('rgb.png','rgb.txt')
                os.system(str)
                os.system(str2)
                tmp+=1


if __name__ == '__main__':                
    quater_test()