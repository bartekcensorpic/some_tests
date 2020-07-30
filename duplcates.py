import pandas as pd




if __name__ == '__main__':

    df = pd.read_csv(r"C:\Users\barte\Documents\Projects\some_tests\detrecks_program\train.csv")

    df =df.drop_duplicates(keep=False,subset=["image_path"], inplace=False)

    df.to_csv(r"C:\Users\barte\Documents\Projects\some_tests\detrecks_program\CorrectLabelsFinal.csv", index=False, quotechar='"', encoding='ascii')

    debug = 5