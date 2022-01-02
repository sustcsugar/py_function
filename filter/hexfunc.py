#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt
import struct

__version__ = '0.1'

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
    sp = img_gray.shape

    for i in range(sp[0]):
        print(i)
        for j in range(sp[1]):
            outfile.write(str('{:02X}'.format(img_gray[i,j]))+'\n')
            #print("pixel out:["+i+" , "+j+"]\n")
    outfile.close()

def hex2image(hex,width,height):
    '''
    将hex文件转为图片
    :param hex: hex文件的路径    
    :return: img_out
    '''
    img_out = np.zeros((height,width,1),np.uint8)
    file_in = open(hex,'r')
    img_hex = file_in.readlines()
   
    data_set = []
    for data in img_hex:
        #data1 = data.strip('\n')
        #data2 = data1.split('\t')
        data1 = data[3:5]
        if data1 == 'xx':
            data1 = str(00)
        data_set.append(data1)
    
    for row in range(height):
        for col in range(width):
            if row*width+col < len(data_set):
                img_out[row,col] = int(data_set[row*width+col],base=16)

    return  img_out 

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



def read_decoder_hexfile(file_name):
    '''
    将decoder文件转为图片
    :param hex: hex文件的路径    
    :return: img_out
    file_name = "sim.dat"
    '''
    file_in = open(file_name,'r')
    hex_line = file_in.readlines()
    img_width = int(hex_line[0])
    img_height = int(hex_line[1])
    img_out = np.zeros((img_width,img_height,3),np.uint8)
    
    for i in range(img_height):
        for j in range(img_width):
            r = int(hex_line[i*img_width+j+2][0:2],base=16)
            b = int(hex_line[i*img_width+j+2][4:6],base=16)
            g = int(hex_line[i*img_width+j+2][2:4],base=16)
            img_out[i,j,0] = b;
            img_out[i,j,1] = g;
            img_out[i,j,2] = r;





