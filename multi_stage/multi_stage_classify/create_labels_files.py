# -*-coding:utf-8-*-

import os,shutil
from map.jd_map import chartolabel


def pre_data(image_path, tar_path):
    if not os.path.exists(tar_path):
        os.mkdir(tar_path)
    train_images = os.listdir(image_path)
    for image in train_images:
        label = image.split('_')[3].split('.')[0]
        print(image, label)
        new_path = os.path.join(tar_path, str(chartolabel[label]))
        print(new_path)
        if not os.path.exists(new_path):
            os.mkdir(new_path)
        shutil.copyfile(os.path.join(image_path, image), os.path.join(new_path, image))

def write_txt(file_dir, txt):
    with open(txt, 'w') as f:
        dirs = os.listdir(file_dir)
        for dir in dirs:
            print(dir)
            if dir == txt.split('/')[-1]:
                break
            else:
                files = os.listdir(os.path.join(file_dir,dir))
                for file in files:
                    f.writelines(dir + '/' + file + ' ' + str(dir) + '\n')


if __name__ == '__main__':
    train_dir = "dataset/jd/jd_train"
    train_tar_dir = "dataset/jd/train"
    train_txt = "dataset/jd/train/train.txt"
    pre_data(train_dir,train_tar_dir)
    write_txt(train_tar_dir, train_txt)

    val_dir = "dataset/jd/jd_val"
    val_tar_dir = "dataset/jd/val"
    val_txt = "dataset/jd/val/val.txt"
    pre_data(val_dir,val_tar_dir)
    write_txt(val_tar_dir,val_txt)

