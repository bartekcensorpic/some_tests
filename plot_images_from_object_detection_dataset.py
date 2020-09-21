import cv2
import random
import numpy as np
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET
import os
from pathlib import Path
from tqdm import tqdm


def convert_annotation(xml_file_path):

    in_file = open(xml_file_path)
    tree = ET.parse(in_file)
    root = tree.getroot()

    classes = []
    boxes = []

    for obj in root.iter("object"):
        cls = obj.find("name").text
        xmlbox = obj.find("bndbox")
        b = (
            int(xmlbox.find("xmin").text),
            int(xmlbox.find("ymin").text),
            int(xmlbox.find("xmax").text),
            int(xmlbox.find("ymax").text),
        )
        class_name = str(cls)
        boxes.append([a for a in b])
        classes.append(class_name)

    return classes, boxes


def cv_plot_bbox(
    img,
    bboxes,
    class_names=None,
    color_class_map=None,
    scale=1.0,
    linewidth=2,
):
    """
    Visualize bounding boxes with OpenCV.
    """

    colors = dict()
    for i, bbox in enumerate(bboxes):
        class_name = class_names[i]

        xmin, ymin, xmax, ymax = [int(x) for x in bbox]

        if (color_class_map is not None) and (class_name.lower() in color_class_map):
            temp = color_class_map[class_name.lower()]
            bcolor = [x for x in temp]
        else:
            colors[class_name] = (random.random(), random.random(), random.random())
            bcolor = [x * 255 for x in colors[class_name]]

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), bcolor, linewidth)

        if class_name:
            y = ymin - 15 if ymin - 15 > 15 else ymin + 15
            cv2.putText(
                img,
                "{:s}".format(class_name),
                (xmin, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                min(scale / 2, 2),
                bcolor,
                min(int(scale), 5),
                lineType=cv2.LINE_AA,
            )
        else:
            #raise ValueError
            print('Missing class name for a bounding box')

    return img


def process_img(img_path, annot_path, save_path,color_class_map):

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    classes, bboxes = convert_annotation(annot_path)
    bboxes = np.asarray(bboxes)
    labels = np.asarray(classes)
    bbox_img = cv_plot_bbox(
        img, bboxes, class_names=labels, color_class_map=color_class_map
    )

    file_save_path = os.path.join(save_path, os.path.basename(img_path))

    bbox_img = cv2.cvtColor(bbox_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_save_path, bbox_img)


if __name__ == "__main__":

    SAVE_FOLDER = r"/mnt/efs/batches_from_BPO/Batch_2_07_09_2020/plots"
    IMG_FOLDER = r"/mnt/efs/raw/img/"
    ANNOTS_FOLDER = r"/mnt/efs/batches_from_BPO/Batch_2_07_09_2020/annots"



    color_class_map = {
        "boobs/pecs": (235, 12, 56),
        "nipples": (235, 12, 187),
        "faces": (68, 12, 235),
        "provocative": (12, 146, 235),
        "vaginas": (12, 235, 198),
        "vagina": (12, 235, 198),
        "naked woman": (12, 235, 127),
        "naked man": (172, 235, 12),
        "naked person": (172, 235, 12),
        "bum": (235, 168, 12),
        "penises": (255, 255, 255),
        "penis": (255, 255, 255),
    }

    xml_files = list(Path(ANNOTS_FOLDER).glob("*.xml"))

    MAX_FILES = len(xml_files)

    annots_list = random.sample(xml_files, MAX_FILES)

    for idx, annot_file_path in tqdm(enumerate(annots_list)):

        annot_file_path = str(annot_file_path)
        annot_file_path = os.path.basename(annot_file_path)
        if idx > MAX_FILES:
            print("MAX FILE LIMIT REACHED, BREAKING LOOP")
            break
        full_annot_path = os.path.join(ANNOTS_FOLDER, annot_file_path)

        full_image_path = os.path.join(IMG_FOLDER, annot_file_path[:-3] + "jpg")
        if os.path.exists(full_image_path):

            try:
                process_img(full_image_path, full_annot_path, SAVE_FOLDER,color_class_map)
            except Exception as e:
                print("ERROR:", annot_file_path)
                print(e)
                continue

        else:
            # todo if we really want, we might just read extension from xml file but lets ignore this for now
            print(
                f"[INFO] image {full_image_path}, does not exist, might be missing file or wrong extension /shrug"
            )
