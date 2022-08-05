# coding: utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import lmdb  # install lmdb by "pip install lmdb"
import cv2
import numpy as np
from glob import glob
import sys
import shutil
import six
from PIL import Image


def checkImageIsValid(imageBin):
    """检验该图片是否有效"""
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)  # 将字符（包含二进制）格式转化为dtype指定格式
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)  # 从内存缓存中，以灰度图格式读取图片
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    """将cache词典中的内容以key-value格式写入lmdb文件"""
    with env.begin(write=True) as txn:
        for k, v in cache.iteritems():
            txn.put(k, v)


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path 。LMDB输出的路径名
        imagePathList : list of image path 。图像路径名组成的list
        labelList     : list of corresponding groundtruth texts 。对应的标签组成的list
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert (len(imagePathList) == len(labelList))  # 检验图像路径名数量与标签数量是否相等
    nSamples = len(imagePathList)  # 图像样本数
    # nSamples = 20000  # 图像样本数

    env = lmdb.open(outputPath, map_size=1099511627776)  # 在该路径下生成空的data.mdb和lock.mdb文件，map_size指定最大容量（kb）
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'r') as f:  # 以二进制读取图片
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt  # %09d表示总共9位，不足部分用0补，如cnt为1，则000000001
        labelKey = 'label-%09d' % cnt
        if cnt%100 == 0:
            print(cnt)
        cache[imageKey] = imageBin
        cache[labelKey] = label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:  # 每1000个样本，将cache写入env一次
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt - 1  # 最终的样本数（因为可能存在无效的样本）
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)  # 最后一行写入"num-sample':str(nasample)，即样本总数
    print('Created dataset with %d samples' % nSamples)


def write_lmdb():
    """给出图片路径和标签，创建lmdb文件"""
    # types = ['wiki', 'weibo', 'sina', 'qqmail', 'ms', 'jd_white', 'jd_grey', 'jd', 'baidu_red', 'baidu_blue', 'baidu',
    #          'apple', 'alipay', '360_gray', '360']
    types = ['baidu_red', 'baidu_blue', 'baidu']
    # types = ['renmin','baidu_Chinese','it168','douban','dajie']  # 表示要生成lmdb的验证码方案，这里只写了'jd'这一示例
    for k in types:
        for kk in ['test']:  # 为每一种验证码的哪些数据集要生成lmdb文件，就写哪些

            outputPath = "/home/abc/LAB_workspace/DAN/Decoupled-attention-network/data/captcha/{0}/{1}_lmdb".format(k, kk)
            if not os.path.exists(outputPath):
                os.mkdir(outputPath)
            else:
                if os.listdir(outputPath):
                    shutil.rmtree(outputPath)
                    # os.mkdir(outputPath)dajie/
            imgDir = "/home/abc/xcx/captcha/{0}/{1}/".format(k,kk)
            imgPathList = glob(os.path.join(imgDir, "*"))
            labelList = []
            for i in range(len(imgPathList)):
                name = os.path.basename(imgPathList[i]).split('_')[1].split('.')[0]

                labelList.append(name.strip())
            print(len(labelList) == len(imgPathList))

            createDataset(outputPath, imgPathList, labelList, lexiconList=None, checkValid=True)


def read_lmdb():
    """给出lmdb文件（data.mdb和lock.mdb）所在路径名，读取里面的内容"""
    path_ldb = "/home/abc/LAB_workspace/DAN/Decoupled-attention-network/data/captcha/360/8000fine/train_lmdb"
    env = lmdb.open(
        path=path_lmdb,  # lmdb存储路径
        max_readers=1,  # 同时读的最大事务数。
        readonly=True,  # 如果为True，则禁止任何写操作
        lock=False,     # 如果为False，则不要执行任何locking操作。
        readahead=False,  # 如果为False, LMDB将禁用OS文件系统的预读机制
        meminit=False)  # 如果为False, LMDB将不会在将缓冲区写入磁盘之前对它们进行零初始化。
    with env.begin(write=False) as txn:  # 读取
        for index in range(1,8501):
            img_key = 'image-%09d' % index
            imgbuf = txn.get(key=img_key.encode())  # 先将key由字符串编码为字节格式，然后根据key获得相应的value，即图像数据
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            img = Image.open(buf)  # 打开图片，PIL格式，二维
            # img.show()
            label_key = 'label-%09d' % index
            label = str(txn.get(label_key.encode()))  # 获取标签，str
            img.save("/home/abc/xcx/CRNN/CRNNdata/360/train/{0}_{1}.png".format(index-1,label))

            print(index)



if __name__ == '__main__':
    write_lmdb()


