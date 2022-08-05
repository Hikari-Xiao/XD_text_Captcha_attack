# coding:utf-8
"""
该代码用于数据增强
"""
import numpy as np
import random
import os
from glob import glob
from PIL import Image, ImageEnhance
from skimage import io, util


def randomColor(image):
    """色彩抖动"""
    r = random.choice([0, 1, 2, 3])
    if r == 0:
        random_factor = np.random.randint(1, 21) / 10.  # 随机因子
        image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    elif r == 1:
        random_factor = np.random.randint(7, 14) / 10.  # 随机因子
        image = ImageEnhance.Brightness(image).enhance(random_factor)  # 调整图像的亮度
    elif r == 2:
        random_factor = np.random.randint(10, 21) / 10.  # 随机因1子
        image = ImageEnhance.Contrast(image).enhance(random_factor)  # 调整图像对比度
    else:
        random_factor = np.random.randint(1, 21) / 10.  # 随机因子
        ImageEnhance.Sharpness(image).enhance(random_factor)  # 调整图像锐度
    return image


def noise(imgPath, save_path):
    img = io.imread(imgPath)

    img = util.random_noise(img, mode='gaussian', seed=None, clip=True, mean=0, var=0.005)
    io.imsave(save_path, img)


def toRGB(imgdir):
    """首先将所有图像转化为rgb格式"""
    imgList = glob(os.path.join(imgdir, '*'))
    for i in range(len(imgList)):
        print(imgList[i])

        img = Image.open(imgList[i])
        img = img.convert('RGB')
        img.save(imgList[i])


if __name__ == "__main__":
    imgdir1 = "data/captcha/ms/train500_enhance"
    imgdir = "data/captcha/ms/train500"
    # toRGB(imgdir)
    imgList = glob(os.path.join(imgdir, '*'))
    for i in range(len(imgList)):
        print(imgList[i])

        img = Image.open(imgList[i])
        # for n in range(6):  # 亮度、对比度等的增强
        #     img = randomColor(img)
        #     save_path = os.path.join(imgdir1, 'enhance'+str(n)+os.path.basename(imgList[i]))
        #     img.save(save_path)

        # for n in range(1):  # 加噪
        #     save_path = os.path.join(imgdir1, 'noise'+str(n)+os.path.basename(imgList[i]))
        #     noise(imgList[i], save_path)

        # for n in range(1):  # 调整尺寸
        #     save_path = os.path.join(imgdir1, 'resize' + str(n) + os.path.basename(imgList[i]))
        #     img = img.resize((256, 64), resample=Image.BILINEAR)
        #     img.save(save_path)
