import glob
import json
import os
from tqdm import tqdm
import cv2

from src.create_annotations import (create_image_annotation, create_annotation_format, find_contours,
                                    get_coco_json_format, create_category_annotation)




# Get "images" and "annotations" info
def images_annotations_info(maskpath):
    annotation_id = 0
    annotations = []
    images = []

    for category in category_ids.keys():
        for mask_image in tqdm(glob.glob(os.path.join(maskpath, category, f'*.{MASK_EXT}'))):
            original_file_name = f'{os.path.basename(mask_image).split(".")[0]}.{ORIGINAL_EXT}'
            mask_image_open = cv2.imread(mask_image)
            height, width, c = mask_image_open.shape

            if original_file_name not in map(lambda img: img['file_name'], images):
                image = create_image_annotation(file_name=original_file_name, width=width, height=height)
                images.append(image)
            else:
                image = [element for element in images if element['file_name'] == original_file_name][0]

            contours = find_contours(mask_image_open)

            for contour in contours:
                annotation = create_annotation_format(contour, image['id'], category_ids[category], annotation_id)
                if annotation['area'] > 0:
                    annotations.append(annotation)
                    annotation_id += 1

    return images, annotations, annotation_id


if __name__ == "__main__":
    # Label ids of the dataset
    category_ids = {
        "chip": 1  # 对应文件夹名称和类别名称
    }

    MASK_EXT = 'tif'
    ORIGINAL_EXT = 'tif'
    coco_format = get_coco_json_format()  # Get the standard COCO JSON format
    
    # 训练mask的json生成
    train_mask_path = './train_mask' # 后面还有chip的文件夹
    # Create category section
    coco_format["categories"] = create_category_annotation(category_ids)
    # Create images and annotations sections
    coco_format["images"], coco_format["annotations"], annotation_cnt = images_annotations_info(train_mask_path)
    with open(f"train_before.json", "w") as outfile:
        json.dump(coco_format, outfile, sort_keys=True, indent=4)
    print("Created %d annotations for images in folder: %s" % (annotation_cnt, train_mask_path))
    
    # 验证集mask的json生成
    val_mask_path = './val_mask' # 后面还有chip的文件夹
    # Create category section
    coco_format["categories"] = create_category_annotation(category_ids)
    # Create images and annotations sections
    coco_format["images"], coco_format["annotations"], annotation_cnt = images_annotations_info(val_mask_path)
    with open(f"val_before.json", "w") as outfile:
        json.dump(coco_format, outfile, sort_keys=True, indent=4)
    print("Created %d annotations for images in folder: %s" % (annotation_cnt, val_mask_path))
