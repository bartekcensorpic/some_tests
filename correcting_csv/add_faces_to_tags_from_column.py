import cv2
import random
import numpy as np
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm



if __name__ == '__main__':

    CSV_FILE =r"C:\Users\barte\Google Drive\10k-dataset-csvs\2.LABELS_FROM_AVERY_WITH_TAGS_TO_COLUMNS_WITH_FACES_UPDATED.csv"

    SAVE_PATH =r"C:\Users\barte\Downloads\LABELS_FROM_AVERY_WITH_TAGS_TO_COLUMNS_WITH_FACES_UPDATED.csv"

    annots_folder = r''


    df = pd.read_csv(CSV_FILE)


    debug = 5







   #df.to_csv(SAVE_PATH, index=False, quotechar='"', encoding='ascii')
    debug = 5
    print('done')
