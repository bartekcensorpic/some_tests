import pandas as pd
import numpy as np
import os


def main():
    csv_path = r"C:\Users\barte\Google Drive\10k-dataset-csvs\2.LABELS_FROM_AVERY_WITH_TAGS_TO_COLUMNS_WITH_FACES_UPDATED.csv"
    save_path =r'C:\Users\barte\Google Drive\10k-dataset-csvs'
    main_df = pd.read_csv(csv_path)

    train, validate, test = np.split(main_df.sample(frac=1), [int(.8 * len(main_df)), int(.9 * len(main_df))])

    ls = {'train':train,
          'valid':validate,
          'test':test}

    for name, df in ls.items():
        new_save_path = os.path.join(save_path,f'{name}_nude.csv')

        df.to_csv(new_save_path, index=False, quotechar='"', encoding='ascii')

    print('done')


if __name__ == '__main__':
    main()