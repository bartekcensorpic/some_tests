import pandas as pd
import os
from correcting_csv.utils import   process_files_generator
import numpy as np

def main():

    folder = r'/mnt/efs/classification_csv/faces_csv/'
    csv_files_paths = process_files_generator(lambda x: x, folder, ['*.csv'], 'getting csvs')

    dfs = []

    for csv_path in csv_files_paths:
        df = pd.read_csv(csv_path)
        dfs.append(df)

    main_df = pd.concat(dfs)

    train, validate, test = np.split(main_df.sample(frac=1), [int(.8 * len(main_df)), int(.9 * len(main_df))])

    save_path = r'/mnt/efs/classification_csv/faces_csv/all_combined.csv'
    main_df.to_csv(save_path, index=False, quotechar='"', encoding='ascii')


    ls = {'train':train,
          'valid':validate,
          'test':test}

    for name, df in ls.items():
        new_save_path = os.path.join(r'/mnt/efs/classification_csv/faces_csv/',f'{name}_faces.csv')

        df.to_csv(new_save_path, index=False, quotechar='"', encoding='ascii')

    print('done')

if __name__ == '__main__':
    main()