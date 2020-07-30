import xml.etree.ElementTree as ET
import os
from pathlib import Path
from tqdm import tqdm

def test_file(file_path):
    in_file = open(file_path)
    tree=ET.parse(in_file)
    root = tree.getroot()


if __name__ == '__main__':
    xml_path = r'/mnt/efs/augmented_v2/annots'
    output = r'/home/ubuntu/Desktop/Trash/empty_annots.txt'



    list_of_xml = list(Path(xml_path).glob('*.xml'))

    for xml in list_of_xml:
        xml = str(xml)
        try:
            test_file(xml)
        except:
            print(xml)
            with open(output, 'a+') as file:
                file.write(xml)

    print('done')