from correcting_csv.utils import process_files_generator
import os
import shutil
if __name__ == '__main__':

    source_folder =  r'/home/ubuntu/Downloads/faces2/lfw-deepfunneled/lfw-deepfunneled'
    destination_folder = r'/mnt/efs/external_datasets/faces'
    gen = process_files_generator(lambda image_path: str(image_path),source_folder, ['*.jpg'],'asd')

    list_of_images = list(gen)


    for source_image_path in list_of_images:
        base_name = os.path.basename(source_image_path)
        destination_image_path = os.path.join(destination_folder, base_name)

        shutil.copy(source_image_path, destination_folder)
        debug = 5

