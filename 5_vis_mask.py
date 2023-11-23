import cv2
import random
import json, os
from pycocotools.coco import COCO
from pycocotools import mask
from skimage import io
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import os.path as osp
from shutil import rmtree
 
def mk_file(file_path: str):
    if osp.exists(file_path):
        rmtree(file_path)
    os.makedirs(file_path)
 
def visualization_bbox_seg(num_image, json_path, save_folder, *str):# 需要画图的是第num副图片， 对应的json路径和图片路径
    mk_file(save_folder)
    
    coco = COCO(json_path)
 
    if len(str) == 0:
        catIds = []
    else:
        catIds = coco.getCatIds(catNms = [str[0]])  # 获取给定类别对应的id 的dict（单个内嵌字典的类别[{}]）
        catIds = coco.loadCats(catIds)[0]['id'] # 获取给定类别对应的id 的dict中的具体id
 
    list_imgIds = coco.getImgIds(catIds=catIds ) # 获取含有该给定类别的所有图片的id
    for img_id in tqdm(list_imgIds):
        img = coco.loadImgs(img_id)[0]  # 获取满足上述要求，并给定显示第num幅image对应的dict
        
        image_name =  img['file_name'] # 读取图像名字
        print(image_name)
        image_id = img['id'] # 读取图像id
        image_info = coco.loadImgs(image_id)[0]
        mask_image = np.zeros((image_info['height'], image_info['width']))
        img_annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None) # 读取这张图片的所有seg_id
        
        img_anns = coco.loadAnns(img_annIds)
        
        for annotation in img_anns:
            rle = mask.frPyObjects(annotation['segmentation'], image_info['height'], image_info['width'])
            m = mask.decode(rle).squeeze()
            mask_image += m  # 这种直接相加得到的mask不准确，尤其是有闭区域边界
        # 将 mask 映射到 [0, 255] 范围
        mask_image = (mask_image * 255).astype(np.uint8)
        
        cv2.imwrite(save_folder + '/' + image_name.split('.')[0] + '.png', mask_image)
    
if __name__ == "__main__":
   visualization_bbox_seg(1, './val.json', './val_draw_mask') # 最后一个参数不写就是画出一张图中的所有类别
   visualization_bbox_seg(1, './train.json', './train_draw_mask') # 最后一个参数不写就是画出一张图中的所有类别