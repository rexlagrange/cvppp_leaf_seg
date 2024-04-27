import os
import numpy as np
import cv2
import PIL.Image as Image
def show_contour(contours, w,h):
    contour = np.array(contours)
    if len(contours.shape) == 2:
        contour = contour[None, :, :]
    img_rgb = np.zeros((w,h))
    img_rgb = np.uint8(img_rgb)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(img_rgb, contour, -1, (255, 0, 0), 3)
    cv2.imshow("1", img_rgb)
    cv2.waitKey(0)
def countour_overlay(img,segment,size=640):# contours是一个list，img是一个图片的路径,把countour画在图片上
    img=img.resize((size,size))
    img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    #yolo格式的countour转换成cv2格式的countour
    segment = np.array(segment).reshape(-1, 2)*size
    segment = segment.astype(np.int32)
    segment = segment.reshape(-1, 1, 2)
    print(segment)
    cv2.drawContours(img_gray, segment, -1, (255, 0, 0), 3)
    cv2.imshow("1", img_gray)
    cv2.waitKey(0)
    return img_gray
def mask2yolo(mask_path, verbose=False,mode=cv2.CHAIN_APPROX_TC89_KCOS):#把1个mask转换为yolo格式
    labels = []  # 类别列表
    segments = []  # 坐标列表
    # 读取图片
    img_current = Image.open(mask_path)
    mat = np.unique(np.array(img_current))
    # PIL图片转换为numpy数组之后由于没有读取调色盘，所以是没有颜色的得到的是全0是背景，全1是第一个mask以此类推
    num_label = len(mat) - 1
    # 获取不同类别的轮廓
    for label_id in range(1, num_label + 1, 1):
        img_temp = img_current
        img_temp = np.uint8(img_temp)
        img_temp = img_temp.copy()
        img_temp[img_temp != label_id] = 0
        w, h = img_current.size
        contours, _ = cv2.findContours(img_temp, cv2.RETR_EXTERNAL,mode)#自动简化边缘
        if len(contours) != 0:  # 获取到该类别轮廓
            for current in contours:
                contour_temp = list()  # 坐标列表
                current = np.array(current).squeeze(1)
                if verbose:show_contour(current, w, h)
                #current_contours 第一列除以h，第二列除以w
                for idx in range(len(current)):
                    contour_temp.append(current[idx][0] / w)
                    contour_temp.append(current[idx][1] / h)
                labels.append(label_id - 1)
                segments.append(contour_temp)
    img_current.close()
    return segments,labels
def _write2txt(out_file, save_dir,segments,labels=None): 
    # 把segments写入txt文件，每行一个类别，第一个数字是类别，后面是坐标()
    # segments是一个list，每个元素是一个list，是一个轮廓的坐标
    # labels是一个list，每个元素是一个int，是一个轮廓的类别
    if labels is None: # 如果没有类别信息，默认为0
        labels = [0]*len(segments)
    with open(save_dir + '/' + out_file, 'w') as f:
        for ln, bb in zip(labels, segments):
            f.write(str(ln) + " " + " ".join([str(round(a, 6)) for a in bb]) + '\n')
def _read_from_txt(txt_file,autolabel=True):#读取一个txt 返回contour_coords_all,contour_labels_all
    segments = []
    labels = []
    with open(txt_file, 'r') as f:
        label_num = 1
        for line in f:
            line = line.strip().split()
            label = int(line[0])
            coords = [float(a) for a in line[1:]]
            segments.append(coords)
            if autolabel:
                labels.append(label_num)
                label_num += 1
            else:
                labels.append(label)
    return segments,labels

def yolo2mask(txtpath,imgsz=640,save_path=None):
    # yolo 网络设置的imgsz默认值是640，图片会被resize到640*640,这个数值可以改变但是一般要求是32的倍数
    # 但是数据集原来的图片不一定是这个大小，这时可以不要保存（save_path=None），返回mask之后再resize
    # 这个函数有bug但是因为yolo的输出结果里本来就有mask，所以就不调试了
    segments,labels = _read_from_txt(txtpath)
    mask = np.zeros((imgsz,imgsz))
    for segment,label in zip(segments,labels):
        contour = np.array(segment).reshape(-1, 2)
        contour = contour * imgsz
        contour = contour.astype(np.int32)
        cv2.fillPoly(mask, [contour], label)
    if save_path is not None:
        cv2.imwrite(save_path, mask)
    else:#show
        mask = cv2.resize(mask, (imgsz, imgsz))
        cv2.imshow("1", mask)
        cv2.waitKey(0)
        
    return mask



# cvppp数据集转换为yolo格式
# 目录下原来是plant???_label.png，和plant???_rgb.png 运行之后会增加plant???_rgb.txt
def cvppp2yolo(mask_dir): 
    mask_files = os.listdir(mask_dir)
    for mask_file in mask_files:
        if mask_file.endswith("label.png"):
            print("processing", mask_file)
            mask_path = os.path.join(mask_dir, mask_file)
            contour_coords_all, _ = mask2yolo(mask_path)
            _write2txt(mask_file.replace("label","rgb").split(".")[0] + ".txt", mask_dir, contour_coords_all)

def main():
    mask_dir = "/Volumes/Intel2T/data/cvppp/CVPPP2017_LSC/label/A1"
    cvppp2yolo(mask_dir)
    mask_dir = "/Volumes/Intel2T/data/cvppp/CVPPP2017_LSC/label/A2"
    cvppp2yolo(mask_dir)
    mask_dir = "/Volumes/Intel2T/data/cvppp/CVPPP2017_LSC/label/A3"
    cvppp2yolo(mask_dir)
    mask_dir = "/Volumes/Intel2T/data/cvppp/CVPPP2017_LSC/label/A4"
    cvppp2yolo(mask_dir)
    
def _debugA3_1():#A3的mask总是有一定的偏移，debug一下(最后发现是mask2yolo里面w和h弄反了)
    segments,labels=_read_from_txt("/Volumes/Intel2T/data/cvppp/CVPPP2017_LSC/label/A3/plant018_rgb.txt")
    img=Image.open("/Volumes/Intel2T/data/cvppp/CVPPP2017_LSC/train/A3/plant018_rgb.png")
    for segment in segments:
        print(segment)
        countour_overlay(img,segment)
def _debugA3_2():
    segments, _ = mask2yolo('/Volumes/Intel2T/data/cvppp/CVPPP2017_LSC/label/A3/plant018_label.png')
    img=Image.open("/Volumes/Intel2T/data/cvppp/CVPPP2017_LSC/train/A3/plant018_rgb.png")
    for segment in segments:
        print(segment)
        countour_overlay(img,segment)

if __name__ == "__main__":
    pass

    