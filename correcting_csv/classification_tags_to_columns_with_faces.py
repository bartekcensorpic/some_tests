import cv2
import random
import numpy as np
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET
import os
from pathlib import Path
import pandas as pd
from mtcnn.mtcnn import MTCNN
from tqdm import tqdm

CLASSES = ['boobsPecs', 'nipples', 'penis', 'vaginas', 'nakedMan', 'nakedWoman', 'nonNude', 'nude']
IMG_FOLDER = r"C:\Users\barte\Desktop\V1-dataset\nude\non_classified"
detector = MTCNN()


def does_image_has_face(image_name):
    image_path = os.path.join(IMG_FOLDER, image_name)

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (416,416))

    faces = detector.detect_faces(img)

    if len(faces) > 0:
        return True
    else:
        return False

def split_tags(string_tags):
    tags = string_tags.replace('"', '').replace('[', '').replace(']', '').replace(',', '').replace('\'', '')

    tags = tags.lower().split()

    return tags


def appluy_to_row(row):
    debug = 5
    tag_list = split_tags(row[1])
    for class_name in CLASSES:
        class_name = class_name.lower()

        if class_name in tag_list:
            row[class_name] = True
        else:
            row[class_name] = False

    if does_image_has_face(row[0]):
        row['faces'] = True
    else:
        row['faces'] = False

    return row






if __name__ == '__main__':

    CSV_FILE = r"C:\Users\barte\Downloads\CorrectLabels (2).csv"
    SAVE_PATH =r"C:\Users\barte\Downloads\LABELS_FROM_AVERY_WITH_TAGS_TO_COLUMNS_WITH_FACES.csv"


    non_classes = ['image_path', 'tags']
    columns = non_classes + CLASSES
    #there was tag 'NUDE' as well

    df = pd.read_csv(CSV_FILE)

    df = df.drop(df.columns.difference(non_classes), 1)



    #new_df = df.apply(appluy_to_row, axis=1)
    new_df = pd.DataFrame()

    for idx, row in tqdm(df.iterrows()):
        new_row = appluy_to_row(row)
        new_df = new_df.append(new_row)
        debug = 5


    new_df.to_csv(SAVE_PATH, index=False, quotechar='"', encoding='ascii')
    debug = 5
    print('done')
