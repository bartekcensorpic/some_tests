import xml.etree.ElementTree as ET
from collections import Counter
import os

def convert_annotation(xml_file_path):

    in_file = open(xml_file_path)
    tree=ET.parse(in_file)
    root = tree.getroot()

    classes = []

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text

        classes.append(cls)

    return classes

def process_file(xml_file_path):
    classes = convert_annotation(xml_file_path)
    return Counter(classes)








