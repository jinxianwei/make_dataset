import json
import os, cv2
import json
import shutil
import cv2
from tqdm import tqdm
import os.path as osp
from shutil import rmtree

def mk_file(file_path: str):
    if osp.exists(file_path):
        rmtree(file_path)
    os.makedirs(file_path)

def draw_bbox(json_path, outpath, image_path):
    json_file = open(json_path)
    infos = json.load(json_file)
    images = infos["images"]
    annos = infos["annotations"]
    assert len(images) == len(images)
    for i in range(len(images)):
        im_id = images[i]["id"]
        im_path = image_path + "/" + images[i]["file_name"]
        img = cv2.imread(im_path)
        for j in tqdm(range(len(annos))):
            if annos[j]["image_id"] == im_id:
                x, y, w, h = annos[j]["bbox"]
                x, y, w, h = int(x), int(y), int(w), int(h)
                x2, y2 = x + w, y + h
                # object_name = annos[j][""]
                img = cv2.rectangle(img, (x, y), (x2, y2), (0, 0, 255), thickness=2)
                img_name = outpath + "/" + images[i]["file_name"]
                cv2.imwrite(img_name, img)
                # continue
        print(i)

if __name__ == "__main__":
    train_json_path = './train.json'
    train_mask_path = 'train_mask/chip'
    train_visual_output = './train_vis_bbox'
    mk_file(train_visual_output)
    draw_bbox(train_json_path, train_visual_output, train_mask_path)
    
    val_json_path = './val.json'
    val_mask_path = 'val_mask/chip'
    val_visual_output = './val_vis_bbox'
    mk_file(val_visual_output)
    draw_bbox(val_json_path, val_visual_output, val_mask_path)
    
    

