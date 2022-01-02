#!/usr/bin/env python
# -*- coding: utf-8 -*-

import hexfunc
import numpy as np
import cv2
import matplotlib.pyplot as plt
import struct


# 将BF的输出结果显示出来，使用[11:4]的数据，raw图
hex_file = "out.txt"
img_out = hexfunc.hex2image(hex_file,1280,720);
cv2.imshow("test",img_out)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 将BF的输出格式化之后输出
outfile = open("img_out.txt","w")
for i in range(img_out.shape[0]):
     print(i)
     for j in range(img_out.shape[1]):
         outfile.write(str('{:02X}'.format(int(img_out[i,j])))+'\n')
         #print("pixel out:["+i+" , "+j+"]\n")
outfile.close()

#s = '123xxxbcd'
#print(s)
#
#s1 = s.replace('x','0')
#print(s1)



