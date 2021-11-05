import numpy as np
import cv2
import matplotlib.pyplot as plt
import struct


def random_noise(image, noise_number):
    '''
    添加随机噪点（实际上就是随机在图像上将像素点的灰度值变为255即白色）
    :param image: 需要加噪的图片
    :param noise_num: 添加的噪音点数目，一般是上千级别的
    :return: img_noise
    '''
    img_noise = image;
    rows,cols,chn = img_noise.shape

    for i in range(noise_number):
        x = np.random.randint(0,rows)
        y = np.random.randint(0,cols)
        img_noise[x,y,:] = 255
    return img_noise

def img2hex(image,hex_out):
    '''
    将图片转换为hex文件, 如果是彩色图片,会转换为gray图.
    :param image: 需要转换的图片
    :param hex_out: 输出hex文件的文件名
    '''
    outfile = open(hex_out,"w")
    img = cv2.imread(image,1)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    for i in range(480):
        print(i)
        for j in range(720):
            outfile.write(str(hex(img_gray[i,j]))+'\n')
            #print("pixel out:["+i+" , "+j+"]\n")
    outfile.close()

def hex2image(hex):
    '''
    将hex文件转为图片
    :param hex: hex文件的路径    
    :return: img_out
    '''
    img_out = np.zeros((480,720,1),np.uint8)
    file_in = open(hex,'r')
    img_hex = file_in.readlines()
   
    data_set = []
    for data in img_hex:
        data1 = data.strip('\n')
        #data2 = data1.split('\t')
        data_set.append(data1)
    
    for row in range(478): #hex文件中, 末尾几行可能是xx或者空值, 无法转换为十进制数字, 简单起见直接将其省略.
        for col in range(720):
            img_out[row,col] = int(data_set[row*720+col],base=16)
   
    return  img_out

hex = "D:\\ProjectCode\\HDL_design\\3DNR\\simulation\\modelsim\\img_output.txt"
img_out = hex2image(hex);
img_in = cv2.imread("test.png",0)
cv2.imshow('img_in',img_in)
cv2.imshow('img_out',img_out)
cv2.waitKey(0)
cv2.destroyAllWindows()
