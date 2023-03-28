import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import ImageDraw, ImageFont, Image

from nets.yolo import YoloBody
from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image, show_config)
from utils.utils_bbox import DecodeBox, DecodeBoxNP

'''
训练自己的数据集必看注释！
'''
class YOLOv4_tiny(object):
    _defaults = {
        "model_path"        : 'model_data/yolov4_tiny_weights_coco.pth',
        "classes_path"      : 'model_data/coco_classes.txt',
        "anchors_path"      : 'model_data/yolo_anchors.txt',
        "anchors_mask"      : [[3,4,5], [1,2,3]],
        "phi"               : 0,  
        "input_shape"       : [416, 416],
        "confidence"        : 0.5,
        "nms_iou"           : 0.3,
        "letterbox_image"   : False,
        "cuda"              : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化YOLO
    #---------------------------------------------------#
    def __init__(self, model_path=None, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value 
            
        #---------------------------------------------------#
        #   获得种类和先验框的数量
        #---------------------------------------------------#
        self.class_names, self.num_classes  = get_classes(self.classes_path)
        self.anchors, self.num_anchors      = get_anchors(self.anchors_path)
        self.bbox_util                      = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]), self.anchors_mask)

        #---------------------------------------------------#
        #   画框设置不同的颜色
        #---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        
        if model_path:
            self.model_path = model_path

        self.generate()

        show_config(**self._defaults)
        
    #---------------------------------------------------#
    #   生成模型
    #---------------------------------------------------#
    def generate(self, onnx=False):
        #---------------------------------------------------#
        #   建立yolo模型，载入yolo模型的权重
        #---------------------------------------------------#
        self.net    = YoloBody(self.anchors_mask, self.num_classes, self.phi)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect(self, image, crop = False, count = False):
        image_shape = np.array(np.shape(image)[2:4])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        # image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        # image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        # image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(image)
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: 
                return []
            
            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]
            
            top_label = top_label.reshape(-1, 1)
            top_conf = top_conf.reshape(-1, 1)
            
            x1, y1, x2, y2 = np.split(top_boxes, 4, axis=1)
            boxes_xyxy = np.concatenate([y1, x1, y2, x2], axis=1)

            output = np.concatenate((boxes_xyxy, top_conf, top_label), axis=1)

            return output

        
    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou)
                                                    
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                #---------------------------------------------------------#
                #   将图像输入网络当中进行预测！
                #---------------------------------------------------------#
                outputs = self.net(images)
                outputs = self.bbox_util.decode_box(outputs)
                #---------------------------------------------------------#
                #   将预测框进行堆叠，然后进行非极大抑制
                #---------------------------------------------------------#
                results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                            image_shape, self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou)
                            
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time