#!/usr/bin/env python
# -*- coding: utf-8 -*-

def gaussian_kernel(size=3,sigma=1,k=1):
    '''
    根据提供的参数生成高斯模板,大小为 mask*mask
    '''
    if sigma==0:
        sigma = ((size-1) * 0.5 -1)*0.3 + 0.8
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
