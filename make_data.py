import xml.etree.ElementTree as ET
import os
import re
import yaml
import random

with open("data/custom.yaml", "r")as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)
    classes = config["names"]
    train_path = config["train"]
    val_path = config["val"]
    test_path = config["test"]


def convert(size, box):
    dw, dh = 1. / (size[0]), 1. / (size[1])
    x, y = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1
    w, h = box[1] - box[0], box[3] - box[2]
    x, y = x * dw, y * dh
    w, h = w * dw, h * dh
    return (x, y, w, h)


def convert_annotation(xml_path):
    with open(xml_path, "r", encoding='UTF-8') as in_file:
        name = xml_path.split('/')[-2]
        print(xml_path)
        txt_file = re.sub(name, 'labels', re.sub('xml', 'txt', xml_path))
        save_path = re.sub(rf'{name}/.+?.xml', 'labels', xml_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        with open(txt_file, "w+", encoding='UTF-8') as out_file:
            tree = ET.parse(in_file)
            root = tree.getroot()
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)
            for obj in root.iter('object'):
                difficult = 0
                if obj.find('difficult') is not None:
                    difficult = obj.find('difficult').text
                cls = obj.find('name').text
                if cls not in classes or int(difficult) == 1:
                    continue
                cls_id = classes.index(cls)
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                     float(xmlbox.find('ymax').text))
                bb = convert((w, h), b)
                out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def voc2yolo(path):
    xmls = os.listdir(path)
    for xml in xmls:
        xml_path = os.path.join(path, xml)
        if xml.split('.')[-1].lower() == 'xml':
            convert_annotation(xml_path)


def write_file(path, datas):
    with open(path, 'w') as f:
        f.write('\n'.join(datas))


def make_train_val_test(path, train_ratio, val_ratio, test_ratio):
    # 只保留jpg格式的文件名称
    images = [image_path for image_path in os.listdir(path) if image_path.endswith('jpg')]
    images_path = [os.path.join(path, image) for image in images]
    random.shuffle(images_path)
    num = len(images_path)
    # train_data = [:0.8] val_data = [0.8:0.9] test_data = [0.9:]
    train_data = images_path[:round(num * train_ratio)]
    val_data = images_path[round(num * train_ratio):round(num * (train_ratio + val_ratio))]
    test_data = images_path[round(num * (train_ratio + val_ratio)):]
    write_file(train_path, train_data)
    write_file(val_path, val_data)
    write_file(test_path, test_data)


if __name__ == "__main__":
    # vol2yolo
    annotations_path = 'data/Annotations'
    voc2yolo(annotations_path)

    # 生成train val test
    images_path = 'data/images'
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    make_train_val_test(images_path, train_ratio, val_ratio, test_ratio)
