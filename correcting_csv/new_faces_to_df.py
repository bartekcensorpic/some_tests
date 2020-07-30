import pandas as pd
import os
from tqdm import tqdm
from correcting_csv.utils import process_files_generator
from sklearn.utils import shuffle


def process_file(image_path):

    row = {'image_path':image_path,
           'tags':['nonnude', 'faces'],
           'boobspecs':False,
           'nipples':False,
           'penis':False,
           'vaginas':False,
           'nakedman':False,
           'nakedwoman':False,
           'nonnude':True,
           'nude':False,
           'faces':True}

    return row


if __name__ == '__main__':
    source_folder = r'/mnt/efs/external_datasets/faces'

    gen = process_files_generator(process_file, source_folder, ['*.jpg', '*.png'], f'getting faces from {source_folder}')

    list_of_agumented_faces = list(gen)

    df = pd.DataFrame()

    for row in tqdm(list_of_agumented_faces):
        df = df.append(row, ignore_index=True)

    df = shuffle(df)
    df.to_csv(r'/mnt/efs/classification_csv/faces_csv/faces_from_faces_folder.csv', index=False, quotechar='"', encoding='ascii')