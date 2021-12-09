#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

def gaussian_kernel(size=3,sigma=1,k=1):
    '''
    根据提供的参数生成高斯模板,大小为 mask*mask
    '''
    if sigma==0:
        sigma = ((size-1) * 0.5 -1)*0.3 + 0.8  # ??
    X = np.linspace(-k,k,size)
    Y = np.linspace(-k,k,size)
    x,y = np.meshgrid(X,Y)
    x0 = 0
    y0 = 0
    gauss = 1/(2*np.pi*sigma**2) * np.exp(- ((x -x0)**2 + (y - y0)**2)/ (2 * sigma**2))
    return gauss

def gaussian_kernel_normal(size=3,sigma=1,k=1):
    '''
    根据提供的参数生成归一化的高斯模板
    param: size 高斯核的大小
    param: sigma 方差
    param: k 高斯核的边界
    '''
    if sigma==0:
        sigma = ((size-1) * 0.5 -1)*0.3 + 0.8  # ??
    X = np.linspace(-k,k,size)
    Y = np.linspace(-k,k,size)
    x,y = np.meshgrid(X,Y)
    x0 = 0
    y0 = 0
    gauss = 1/(2*np.pi*sigma**2) * np.exp(- ((x -x0)**2 + (y - y0)**2)/ (2 * sigma**2))
    sum = 0
    for i in gauss:
        for j in i:
            sum = sum +j
    normal = gauss/sum

    return normal

def color_level(level=6,sigma_color=1):
    value = int(256/level)
    vector = np.zeros(level)
    for i in range(level):
        vector[i]=(i+1)*value
    color = np.exp( -( (vector**2)/(2*sigma_color**2) ) )
    #sum=0
    #for i in color:
    #    sum = sum + i
    #color = color/sum

    return color

def bilateral_filter(size=3,sigma_color=1,sigma_space=1,k=1):
    '''
    根据提供的参数生成归一化的双边模板
    param: size 核的大小
    param: sigma_color 色域方差
    param: sigma_space 空域方差
    param: k 核边界
    '''
    gaussian = gaussian_kernel_normal(5,1,1)
    color = color_level(6,30)

def read_isp_file(file_name,write_file):
    '''
    读取isp的地址文件
    param: file 输入的文件
    '''
    file_in = open(file_name,'r')
    lines = file_in.readlines()
    write_file.write("//"+file_name+ "\n")
    for line in lines:
        if(len(line) > 100):
            element = line.split(',')
            write_file.write(element[5]+' : '+element[2]+"\n")
            print(element[5]+' : '+element[2]+'\n')
    write_file.write("\n\n")
    
def img_resize(img_name):
    img = cv2.imread(img_name,0)
    cv2.imshow("origin_img",img)
    shrink_img1 = cv2.resize(img,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST)
    shrink_img2 = cv2.resize(img,None,fx=2,fy=2,interpolation=cv2.INTER_LINEAR)
    shrink_img3 = cv2.resize(img,None,fx=2,fy=2,interpolation=cv2.INTER_AREA)
    shrink_img4 = cv2.resize(img,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
    cv2.imshow("shrink_img1",shrink_img1)
    cv2.imshow("shrink_img2",shrink_img2)
    cv2.imshow("shrink_img3",shrink_img3)
    cv2.imshow("shrink_img4",shrink_img4)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

