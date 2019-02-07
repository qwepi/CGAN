# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
import cv2
import pdb
import PIL.Image as img
 
#读取图片的函数，接收六个参数
#输入参数分别是图片名，图片路径，标签路径，图片格式，标签格式，需要调整的尺寸大小
def ImageReader(filename, picture_path, label_path, picture_format = ".png", label_format = ".PNG", size = 256,batchsize = 4):
    #picture_name = picture_path + file_name + picture_format #得到图片名称和路径
    #pdb.set_trace()
    picture_resize_list = []
    label_resize_list = []
    i = 0
    #pdb.set_trace()
    if not batchsize == 1:
        for file_name in filename:
            picture_name = picture_path + file_name + '_layout'+picture_format #得到图片名称和路径
            label_name = label_path + file_name + label_format #得到标签名称和路径
            picture = cv2.imread(picture_name, 1) #读取图片
            label = cv2.imread(label_name, 1) #读取标签
            if i == 0:
                height = picture.shape[0] #得到图片的高
                width = picture.shape[1] #得到图片的宽
                i = 1
            picture_resize_t = cv2.resize(picture, (size, size)) #调整图片的尺寸，改变成网络输入的大小
            picture_resize = picture_resize_t / 127.5 - 1. #归一化图片
            label_resize_t = cv2.resize(label, (size, size)) #调整标签的尺寸，改变成网络输入的大小
            label_resize = label_resize_t / 127.5 - 1. #归一化标签
            picture_resize_list.append(picture_resize)
            label_resize_list.append(label_resize)
    else:
        file_name = filename
        picture_name = picture_path + file_name + '_layout'+picture_format #得到图片名称和路径
        label_name = label_path + file_name + label_format #得到标签名称和路径
        picture = cv2.imread(picture_name, 1) #读取图片
        label = cv2.imread(label_name, 1) #读取标签
        height = picture.shape[0] #得到图片的高
        width = picture.shape[1] #得到图片的宽
        picture_resize_t = cv2.resize(picture, (size, size)) #调整图片的尺寸，改变成网络输入的大小
        picture_resize = picture_resize_t / 127.5 - 1. #归一化图片
        label_resize_t = cv2.resize(label, (size, size)) #调整标签的尺寸，改变成网络输入的大小
        label_resize = label_resize_t / 127.5 - 1. #归一化标签
        picture_resize_list = picture_resize
        label_resize_list = label_resize
    picture_resize_list = np.array(picture_resize_list)
    label_resize_list = np.array(label_resize_list)
    return picture_resize_list, label_resize_list, height, width #返回网络输入的图片，标签，还有原图片和标签的长宽

def ImageReader_list(filename, picture_path, label_path, picture_format = ".png", label_format = ".PNG", size = 256):
    #picture_name = picture_path + file_name + picture_format #得到图片名称和路径
    #pdb.set_trace()
    picture_resize_list = []
    label_resize_list = []
    i = 0
    for file_name in filename:
        if 'b2' in file_name:
            picture_name = picture_path + file_name + '_layout'+picture_format #得到图片名称和路径
            label_name = label_path + file_name + label_format #得到标签名称和路径
            picture = cv2.imread(picture_name, 1) #读取图片
            label = cv2.imread(label_name, 1) #读取标签
            picture_resize_t = cv2.resize(picture, (size, size)) #调整图片的尺寸，改变成网络输入的大小
            label_resize = cv2.resize(label, (size, size)) #调整标签的尺寸，改变成网络输入的大小
            print(file_name)
            cv2.imwrite("./0.png",picture_resize_t)
            #rotate90
            picture_resize_90 = np.rot90(picture_resize_t)
            picture_resize_list.append(picture_resize_90)
            label_resize_list.append(label_resize)
            cv2.imwrite("./90.png",picture_resize_90)
            #rotate180
            picture_resize_180 = np.rot90(picture_resize_t,2)
            picture_resize_list.append(picture_resize_180)
            label_resize_list.append(label_resize)
            cv2.imwrite("./180.png",picture_resize_180)
            #rotate270
            picture_resize_270 = np.rot90(picture_resize_t,3)
            picture_resize_list.append(picture_resize_270)
            label_resize_list.append(label_resize)
            cv2.imwrite("./270.png",picture_resize_270)
            #flip_topdown
            picture_resize_td = picture_resize.transpose(img.FLIP_TOP_BOTTOM)
            picture_resize_list.append(picture_resize_td)
            label_resize_list.append(label_resize)
            cv2.imwrite("./td.png",picture_resize_td)
            #flip_leftright
            picture_resize_lr = picture_resize.transpose(img.FLIP_LEFT_RIGHT)
            picture_resize_list.append(picture_resize_lr)
            label_resize_list.append(label_resize)
            cv2.imwrite("./lr.png",picture_resize_lr)
                       
            picture_resize = picture_resize_t / 127.5 - 1. #归一化图片
            label_resize_t = cv2.resize(label, (size, size)) #调整标签的尺寸，改变成网络输入的大小
            label_resize = label_resize_t / 127.5 - 1. #归一化标签
            picture_resize_list.append(picture_resize)
            label_resize_list.append(label_resize)
            if i == 0:
                break
    return picture_resize_list, label_resize_list #返回网络输入的图片，标签，还有原图片和标签的长宽
