import pandas as pd
import re
import ast
import os
from tqdm import tqdm

def has_to_change_faces(df):
    orginal_len = len(df)
    temp = df[df['faces']==True]
    new_len =len(temp)
    if new_len > 0 and orginal_len != new_len:
        return True
    else:
        return False

def add_faces_to_tags(row):
    if 'faces' not in row['tags']:
        row['tags'].append('faces')
    return row

if __name__ == '__main__':
    '''
    changes faces in CSV per agumentations: if one agumentation has detected faces, then all of them have
    '''

    folder = r'/home/ubuntu/Desktop/Trash/'
    train = r'/home/ubuntu/Desktop/Trash/train.csv'
    val = r'/home/ubuntu/Desktop/Trash/val.csv'
    test = r'/home/ubuntu/Desktop/Trash/test.csv'

    list_df = {'train':train,'val':val,'test': test}

    pattern =  r'(?P<file_path>.*)(_aug_){1}\d+_\d+(\.jpg|\.png)' #we want to group them by their original name. because i wrote it, i know that the pattern is: '<file_path>_aug_<any number>_<any number>.jpg'
    regex =  re.compile(pattern)


    for name_df,df_path in list(list_df.items()):
        source_df = pd.read_csv(df_path,converters={1:ast.literal_eval})
        dest_df = source_df.copy()

        source_df = source_df[source_df['image_path'].str.startswith('/mnt/efs/augmented_v1/negative/', na=False)]

        pbar = tqdm(source_df.iterrows())

        for index, row in pbar:
            pbar.set_description(f'going by {name_df}')
            file_name = row['image_path']
            matches = regex.search(file_name)
            org_path = matches.group('file_path')

            this_rows_augmenations = source_df[source_df['image_path'].str.contains(org_path+'(_aug_){1}\d+_\d+(\.jpg|\.png)')]
            has_to_change = has_to_change_faces(this_rows_augmenations)

            if has_to_change:
                for inner_index, augmented_row in this_rows_augmenations.iterrows():
                    agumented_path = augmented_row['image_path']
                    to_change = dest_df[dest_df['image_path'] == agumented_path]
                    to_change['faces'] = True
                    to_change.apply(add_faces_to_tags, axis=1)

                    dest_df.loc[dest_df['image_path'] == agumented_path] = to_change

                    source_df.drop(inner_index, inplace=True)
                    debug = 5



        ## save destination to file
        dest_df.to_csv(os.path.join(folder, f'{name_df}_aug_faces.csv'), index=False, quotechar='"', encoding='ascii')

    print('done')








