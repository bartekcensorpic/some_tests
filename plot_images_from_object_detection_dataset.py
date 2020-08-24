import cv2
import random
import numpy as np
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET
import os
from pathlib import Path

def convert_annotation(xml_file_path):

    in_file = open(xml_file_path)
    tree=ET.parse(in_file)
    root = tree.getroot()

    classes = []
    boxes = []

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        class_name = str(cls)
        boxes.append([a for a in b])
        classes.append(class_name)

    return classes, boxes

def cv_plot_bbox(img, bboxes, scores=None, labels=None, thresh=0.5,
                 class_names=None, colors=None,
                 absolute_coordinates=True, scale=1.0, linewidth=2):
    """Visualize bounding boxes with OpenCV.

    Parameters
    ----------
    img : numpy.ndarray or mxnet.nd.NDArray
        Image with shape `H, W, 3`.
    bboxes : numpy.ndarray or mxnet.nd.NDArray
        Bounding boxes with shape `N, 4`. Where `N` is the number of boxes.
    scores : numpy.ndarray or mxnet.nd.NDArray, optional
        Confidence scores of the provided `bboxes` with shape `N`.
    labels : numpy.ndarray or mxnet.nd.NDArray, optional
        Class labels of the provided `bboxes` with shape `N`.
    thresh : float, optional, default 0.5
        Display threshold if `scores` is provided. Scores with less than `thresh`
        will be ignored in display, this is visually more elegant if you have
        a large number of bounding boxes with very small scores.
    class_names : list of str, optional
        Description of parameter `class_names`.
    colors : dict, optional
        You can provide desired colors as {0: (255, 0, 0), 1:(0, 255, 0), ...}, otherwise
        random colors will be substituted.
    absolute_coordinates : bool
        If `True`, absolute coordinates will be considered, otherwise coordinates
        are interpreted as in range(0, 1).
    scale : float
        The scale of output image, which may affect the positions of boxes
    linewidth : int, optional, default 2
        Line thickness for bounding boxes.
        Use negative values to fill the bounding boxes.

    Returns
    -------
    numpy.ndarray
        The image with detected results.

    """


    if labels is not None and not len(bboxes) == len(labels):
        raise ValueError('The length of labels and bboxes mismatch, {} vs {}'
                         .format(len(labels), len(bboxes)))
    if scores is not None and not len(bboxes) == len(scores):
        raise ValueError('The length of scores and bboxes mismatch, {} vs {}'
                         .format(len(scores), len(bboxes)))


    # use random colors if None is provided
    if colors is None:
        colors = dict()
    for i, bbox in enumerate(bboxes):
        if scores is not None and scores.flat[i] < thresh:
            continue

        cls_id = -1
        colors[cls_id] = (random.random(), random.random(), random.random())
        xmin, ymin, xmax, ymax = [int(x) for x in bbox]
        bcolor = [x * 255 for x in colors[cls_id]]
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), bcolor, linewidth)

        if class_names is not None and cls_id < len(class_names):
            class_name = class_names[cls_id]
        else:
            class_name = str(cls_id) if cls_id >= 0 else ''
        score = '{:d}%'.format(int(scores.flat[i]*100)) if scores is not None else ''
        if class_name or score:
            y = ymin - 15 if ymin - 15 > 15 else ymin + 15
            cv2.putText(img, '{:s} {:s}'.format(class_name, score),
                        (xmin, y), cv2.FONT_HERSHEY_SIMPLEX, min(scale/2, 2),
                        bcolor, min(int(scale), 5), lineType=cv2.LINE_AA)

    return img

def process_img(img_path,annot_path, save_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    classes, bboxes = convert_annotation(annot_path)
    bboxes = np.asarray(bboxes)
    labels = np.asarray(classes)
    bbox_img = cv_plot_bbox(img,bboxes ,class_names=labels )

    file_save_path = os.path.join(save_path, "test_" +img_path)
    cv2.imwrite(file_save_path, bbox_img)



if __name__ == '__main__':

    SAVE_FOLDER = r"/mnt/efs/batches_from_BPO/Batch_1_24_08_2020/plots"
    IMG_FOLDER = r"/mnt/efs/raw/img/"
    ANNOTS_FOLDER = r"/mnt/efs/batches_from_BPO/Batch_1_24_08_2020/annots"

    MAX_FILES = 1000

    annots_list = random.sample(list(Path(ANNOTS_FOLDER).glob('*.xml')),MAX_FILES)



    for idx,annot_file_path in enumerate(annots_list):

        if idx > MAX_FILES:
            print('MAX FILE LIMIT REACHED, BREAKING LOOP')
            break
        full_annot_path = os.path.join(ANNOTS_FOLDER, annot_file_path)

        full_image_path = os.path.join(IMG_FOLDER, annot_file_path[:-3] + "jpg")
        if os.path.exists(full_image_path):


            try:
                process_img(full_image_path, full_annot_path, SAVE_FOLDER)
            except Exception as e:
                print('ERROR:', annot_file_path)
                continue


        else:
            #todo if we really want, we might just read extention from xml file but lets ignore this now
            print(f"[INFO] image {full_image_path}, does not exist, might be missing file or wrong extention /shrug")








