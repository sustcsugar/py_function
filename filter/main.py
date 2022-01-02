#!/usr/bin/env python
# -*- coding: utf-8 -*-

import hexfunc
import numpy as np
import cv2
import matplotlib.pyplot as plt
import struct


hex_file = "out.hex"
img_out = func.hex2image(hex_file,1280,720);
cv2.imshow("test",img_out)
cv2.waitKey(0)
cv2.destroyAllWindows()

outfile = open("img_out_hex","w")
for i in range(img_out.shape[0]):
     print(i)
     for j in range(img_out.shape[1]):
         outfile.write(str('{:02X}'.format(int(img_out[i,j])))+'\n')
         #print("pixel out:["+i+" , "+j+"]\n")
outfile.close()

