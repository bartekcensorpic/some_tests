import cv2
import random
import numpy as np
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET
import os
from pathlib import Path
import pandas as pd

CLASSES = ['boobsPecs'.lower(), 'nipples'.lower(), 'penis'.lower(), 'vaginas'.lower(), 'nakedMan'.lower(),
           'nakedWoman'.lower(), 'nonNude'.lower(), 'nude'.lower(), 'faces'.lower()]
NUDE_CLASSES = ['boobsPecs'.lower(), 'nipples'.lower(), 'penis'.lower(), 'vaginas'.lower(), 'nakedMan'.lower(),
                'nakedWoman'.lower()]
NUDE = 'nude'.lower()
NON_NUDE = 'nonNude'.lower()


def split_tags(string_tags):
    tags = string_tags.replace('"', '').replace('[', '').replace(']', '').replace(',', '').replace('\'', '')

    tags = tags.lower().split()

    return tags


def appluy_to_row(row):
    tag_list = split_tags(row[1])
    tag_list = list(map(lambda s: s.lower(), tag_list))

    # add nude if any nude tag, non_nude if not
    in_row_tags = set(tag_list)
    all_nude_tags = set(NUDE_CLASSES)

    shared = list(in_row_tags.intersection(all_nude_tags))

    if len(shared) > 0 and NUDE not in tag_list:
        # means there are nudes
        tag_list.append(NUDE)

    elif len(shared) < 0 and NON_NUDE not in tag_list:
        tag_list.append(NON_NUDE)
    else:
        #nothing
        debug = 5

    # mark true/false columns
    for class_name in CLASSES:
        class_name = class_name.lower()

        if class_name in tag_list:
            row[class_name] = True
        else:
            row[class_name] = False

    row['tags'] = tag_list
    return row


if __name__ == '__main__':
    CSV_FILE = r"/home/ubuntu/Desktop/Trash/NUDES_ONLY_WITH_AGUMENTATIONS_NO_NEGATIVES.csv"
    SAVE_PATH = r"/home/ubuntu/Desktop/Trash/CORRECTED_NUDES_ONLY_WITH_AGUMENTATIONS_NO_NEGATIVES.csv"

    non_classes = ['image_path', 'tags']
    columns = non_classes + CLASSES
    # there was tag 'NUDE' as well

    df = pd.read_csv(CSV_FILE)

    #df = df.drop(df.columns.difference(non_classes), 1)

    new_df = df.apply(appluy_to_row, axis=1)

    new_df.to_csv(SAVE_PATH, index=False, quotechar='"', encoding='ascii')
    debug = 5
    print('done')
