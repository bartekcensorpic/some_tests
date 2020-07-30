import pandas as pd


NUDE_COLUMNS = ['boobspecs','nipples','penis','vaginas','nakedman','nakedwoman','nude']
NON_NUDE_COLUMNS = ['nonnude','faces']

BINARY_NUDE = 'BINARY_NUDE'
BINARY_NON_NUDE = 'BINARY_NONNUDE'

def add_binary_column(row):

    row[BINARY_NUDE] = False

    for nude in NUDE_COLUMNS:
        if row[nude]:
            row[BINARY_NUDE] = True

    return row



def process_dataframe(df):

    df = df.apply(add_binary_column, axis=1)

    return df

def main():

    csv_path = '/mnt/efs/classification_csv/shuffled_test.csv'
    csv_output_path = '/mnt/efs/classification_csv/shuffled_test_with_binary.csv'

    df = pd.read_csv(csv_path)

    df = process_dataframe(df)

    df.to_csv(csv_output_path, index=False, quotechar='"', encoding='ascii')





if __name__ == '__main__':
    main()





