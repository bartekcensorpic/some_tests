import numpy as np
import cv2
import os
if __name__ == '__main__':

    data = np.load('/home/ubuntu/Downloads/faces1/face_images.npz')
    output_folder = r'/mnt/efs/external_datasets/faces'
    lst = data.files
    # for item in lst:
    #     print(item)
    #     print(data[item])

    images =  data['face_images']
    for i in range(7049):
        image = images[..., i]

        image_path = os.path.join(output_folder, f'face{i}.jpg')
        cv2.imwrite(image_path, image)
        debug = 5


