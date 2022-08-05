# coding: utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import lmdb  # install lmdb by "pip install lmdb"
import cv2
import numpy as np


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
    assert (len(imagePathList) == len(labelList))      # 检验图像路径名数量与标签数量是否相等
    nSamples = len(imagePathList)  # 图像样本数
    env = lmdb.open(outputPath, map_size=1099511627776)  # 在该路径下生成空的data.mdb和lock.mdb文件，map_size指定最大容量（kb）
    cache = {}
    cnt = 1
    for i in xrange(nSamples):
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

        imageKey = 'image-%09d' % cnt   # %09d表示总共9位，不足部分用0补，如cnt为1，则000000001
        labelKey = 'label-%09d' % cnt
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
    nSamples = cnt - 1   # 最终的样本数（因为可能存在无效的样本）
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)   # 最后一行写入"num-sample':str(nasample)，即样本总数
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    pass
