import json
import os
from tqdm import tqdm

small_area = 40

def filter_area(before_ann_json_path, filtered_ann_json_path):
    with open(before_ann_json_path, 'r') as f:
        ann_dict = json.load(f)
    new_annotations = []
    for i in tqdm(range(len(ann_dict['annotations']))):
        if ann_dict['annotations'][i]['area'] > small_area:
            new_annotations.append(ann_dict['annotations'][i])
    ann_dict['annotations'] = new_annotations
    with open(filtered_ann_json_path, 'w') as f:
        json.dump(ann_dict, f, indent=4)

if __name__ =="__main__":
    filter_area('./train_before.json', './train.json')
    filter_area('./val_before.json', './val.json')