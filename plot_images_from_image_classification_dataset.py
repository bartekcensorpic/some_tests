import os

import cv2
import pandas as pd


def write_labels_on_images(pixels, label):
    bcolor = [x * 255 for x in (255, 0, 0)]
    cv2.putText(pixels, label,
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1/2,
                bcolor, min(int(1), 5), lineType=cv2.LINE_AA)

    return pixels


def process_img(img_path,label_str, save_path, image_name):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    labeled_img = write_labels_on_images(img,label_str)

    file_save_path = os.path.join(save_path, str(image_name)+label_str+'.jpg')
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_save_path, labeled_img)


if __name__ == '__main__':

    SAVE_FOLDER = r"C:\Users\barte\Desktop\trash\tmp"
    IMG_FOLDER = r"C:\Users\barte\Desktop\V1-dataset\nude\non_classified"
    CSV_FILE = r"C:\Users\barte\Downloads\CorrectLabels (2).csv"

    MAX_FILES = 1000

    #annots_list = random.sample(list(Path(ANNOTS_FOLDER).glob('*.xml')),MAX_FILES)

    df = pd.read_csv(CSV_FILE)

    random_df = df.sample(MAX_FILES)


    for idx,row in random_df.iterrows():

        image_name = row[0]
        labels_str = row[1]



        full_image_path = os.path.join(IMG_FOLDER, image_name)
        if os.path.exists(full_image_path):

            try:
                process_img(full_image_path, labels_str, SAVE_FOLDER,idx)
            except Exception as e:
                print('ERROR:', image_name)
                print('idx:', idx)
                continue
        else:
            #todo if we really want, we might just read extention from xml file but lets ignore this now
            print(f"[INFO] image {full_image_path}, does not exist, might be missing file or wrong extention /shrug")





    print('done')


