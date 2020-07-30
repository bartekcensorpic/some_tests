import cv2
import random
import numpy as np
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET
import os
from pathlib import Path
import pandas as pd
import re
from sklearn.utils import shuffle

from correcting_csv.utils import process_files_generator


def get_base_name_without_extentions(file_path):
    img_name_extention = os.path.basename(file_path)
    img_name = os.path.splitext(img_name_extention)[0]

    return img_name

def check_for_faces(element):

    name_list = list(element.iter('name'))
    if len(name_list) > 0:
        name_el = name_list[0]
        if name_el.text.lower() == 'faces' or name_el.text.lower() == 'face':
            return True

    return False


def does_xml_has_face(xml_file_path):

    with open(xml_file_path) as in_file:
        tree = ET.parse(in_file)
        root = tree.getroot()
        root_iterator = list(root.findall("./object"))
        objects_with_faces = [element for element in root_iterator if check_for_faces(element)]

        image_name = get_base_name_without_extentions(xml_file_path) + '.jpg'

        has_faces = True if len(objects_with_faces) > 0 else False


    return image_name, has_faces


def group_by_augmentations(df):
    pattern =  r'(?P<file_path>.*)(_aug_){1}\d+_\d+(\.jpg|\.png)' #we want to group them by their original name. because i wrote it, i know that the pattern is: '<file_path>_aug_<any number>_<any number>.jpg'
    regex =  re.compile(pattern)

    image_to_augmentation_dict = dict()
    for idx,row in df.iterrows():

        file_name = row['image_path']
        matches = regex.search(file_name)
        org_path = matches.group('file_path')

        if org_path not in image_to_augmentation_dict:
            image_to_augmentation_dict[org_path] = [row]
        else:
            image_to_augmentation_dict[org_path].append(row)

    return image_to_augmentation_dict

def split_and_shufle(group_aug_dict, columns):
    data = [ (random.random(), group) for group in group_aug_dict.items() ]
    data.sort()
    n_lines = len(data)
    n_train = int(n_lines * 0.6)
    rest =  int(n_lines - n_train)
    n_test = int(rest /2)

    print('train_df')
    train_df = group_aug_to_df( data[:n_train],columns)
    print('test_df')
    test_df = group_aug_to_df( data[n_train:n_train+n_test],columns)
    print('valid_df')
    valid_df = group_aug_to_df( data[n_train+n_test:],columns)

    return train_df, valid_df, test_df

def group_aug_to_df(data, columns):

    df =  pd.DataFrame(columns=columns)
    for random_number, group in data:
        for row in group[1]:
                df.loc[len(df)] = row

    return df

if __name__ == '__main__':

    CSV_FILE =r"/home/ubuntu/Desktop/Trash/CORRECTED_NUDES_ONLY_WITH_AGUMENTATIONS_NO_NEGATIVES.csv"
    SAVE_PATH =r"/home/ubuntu/Desktop/Trash/"
    XML_FOLDER = r'/mnt/efs/augmented_v1/negative_face_annots'
    img_folder = r'/mnt/efs/augmented_v1/negative'


    df = pd.read_csv(CSV_FILE)

    columns = list(df.columns)

    generator_result = process_files_generator(does_xml_has_face, XML_FOLDER, ['*.xml'], 'collecting xml files')

    #generator_result = list(generator_result)
    for image_name, has_face in generator_result:
        row = {'image_path': os.path.join(img_folder,image_name),
               'tags': ['nonnude','faces'] if has_face else ['nonnude'],
               'boobspecs': False,
               'nipples': False,
               'penis': False,
               'vaginas': False,
               'nakedman': False,
               'nakedwoman': False,
               'nonnude': True,
               'nude': False,
               'faces': True if has_face else False,
               }
        df = df.append(row,ignore_index=True)

    df.to_csv(os.path.join(SAVE_PATH, 'FULL_DATASET.csv') , index=False, quotechar='"', encoding='ascii')

    print('aumentations')
    group_aug_dict = group_by_augmentations(df)
    train_df, validation_df, test_df = split_and_shufle(group_aug_dict, columns)

    ls = {'train.csv': train_df,
          'val.csv': validation_df,
          'test.csv':test_df}

    for name, df in ls.items():
        save_path = os.path.join(SAVE_PATH, name)
        df = shuffle(df)
        df.to_csv(save_path, index=False, quotechar='"', encoding='ascii')

    debug = 5
    print('done')
