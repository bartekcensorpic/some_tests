from PIL import Image
from pathlib import Path
from tqdm import tqdm

if __name__ == '__main__':
    FOLDER_PATH = r'/mnt/efs/raw/blurred_sharp/test_output/sharp/'

    files = []
    for ex in ['*.jpg', '*.png']:
        files.extend(Path(FOLDER_PATH).rglob(ex))

    pbar = tqdm(files)

    print('n images:', len(files))

    for file_path in pbar:
        try:
            image = Image.open(file_path)
        except Exception as e:
            print('#############################')
            print(file_path)
            print(e)

    print('done')
