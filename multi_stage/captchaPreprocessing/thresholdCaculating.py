import cv2 as cv
import numpy as np
import math

'''
Binarization  Integration
Integrate different threshold selection methods 
将各种二值化不同阈值选取的方法整合在一起
'''

# input: the image of the captcha
# output: the threshold of binarization

# Percentage threshold method,
# which can better retain the noise line
# 百分比阈值法，能较好地保留噪线，返回每张图片的二值化阈值
def GetPTileThreshold(path, gray):
    if(gray == "True"):
        image = cv.imread(path)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        size = gray.shape[0] * gray.shape[1]
        HistGram = gray.ravel()
    else:
        image = cv.imread(path, 0)
        size = image.shape[0] * image.shape[1]
        HistGram = image.ravel()
    HistGram_1D = [0] * 256
    for i in range(size):
        index = HistGram[i]
        HistGram_1D[index] = HistGram_1D[index] + 1
    amount = 0
    sum = 0
    for i in range(0,256):
        amount = amount + HistGram_1D[i]
    for i in range(0,256):
        sum = sum + HistGram_1D[i]
        if sum >= (amount / 10):
            return i
    return -1

# Based on calculating the average of gray
# 基于灰度平均值的阈值
def average_threshold(image):
    # gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    size = image.shape[0] * image.shape[1]
    HistGram = image.ravel()
    HistGram_1D = [0] * 256
    for i in range(size):
        index = HistGram[i]
        HistGram_1D[index] = HistGram_1D[index] + 1
    sum = 0
    amount = 0
    for i in range(0,256):
        amount = amount + HistGram_1D[i]
        sum = sum + i * HistGram_1D[i]
    return sum/amount

# Iterative method for threshold
# 迭代阈值法
def Iterative_best_threshold(image, gray):
    if(gray == "True"):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        size = gray.shape[0] * gray.shape[1]
        HistGram = gray.ravel()
    else:
        size = image.shape[0] * image.shape[1]
        HistGram = image.ravel()
    HistGram_1D = [0] * 256
    for i in range(size):
        index = HistGram[i]
        HistGram_1D[index] = HistGram_1D[index] + 1
    minvalue = 0
    maxvalue = 255
    for minvalue in range(0, 255):
        if HistGram_1D[minvalue] == 0:
            continue
        else:
            break
    for maxvalue in range(255, minvalue, -1):
        if HistGram_1D[maxvalue] == 0:
            continue
        else:
            break
    if minvalue == maxvalue:
        return maxvalue
    if minvalue + 1 == maxvalue:
        return minvalue
    print(minvalue, maxvalue)
    threshold = minvalue
    newthreshold = (int(minvalue + maxvalue)) >> 1
    # Stop iterating when the current and last two thresholds are the same
    # 当前后两次阈值相同时停止迭代
    Iter = 0
    while(threshold != newthreshold):
        Sum_one = 0
        Sum_Integer_one = 0
        Sum_two = 0
        Sum_Integer_two = 0
        threshold = newthreshold
        # 将图像分为前景和背景两部分，求出两部分的平均值
        for i in range(minvalue,threshold+1):
            Sum_Integer_one = Sum_Integer_one + HistGram_1D[i] * i
            Sum_one = Sum_one + HistGram_1D[i]
        meanvalue_one = Sum_Integer_one / Sum_one
        for i in range(threshold+1,maxvalue+1):
            Sum_Integer_two = Sum_Integer_two + HistGram_1D[i] * i
            Sum_two = Sum_two + HistGram_1D[i]
        meanvalue_two = Sum_Integer_two / Sum_two
        newthreshold = (int(meanvalue_one + meanvalue_two)) >> 1
        Iter = Iter + 1
        if Iter >= 1000:
            return -1
    return threshold

# One-dimensional maximum entropy threshold method
# Method idea: Given a threshold q, divide the image into P0 and P1 (foreground and background),
# calculate the probability of each gray value, and then calculate and sum the corresponding entropy of the foreground and background,
# traverse 0-255 to find Maximum entropy.
# which can accurately retain the noise lines
# 一维最大熵阈值法
# 方法思想：给定阈值q将图像分割为P0和P1（前景和背景），分别计算出每一灰度值出现的概率，再计算前景和背景对应的熵并求和，遍历0-255找出最大熵
# 一维最大熵二值化能准确保留噪线
def MaxEntropy_1D(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    size = gray.shape[0] * gray.shape[1]
    HistGram = gray.ravel()
    HistGram_1D = [0] * 256
    for i in range(size):
        index = HistGram[i]
        HistGram_1D[index] = HistGram_1D[index] + 1
    minvalue = 0
    maxvalue = 255
    for minvalue in range(0,255):
        if HistGram_1D[minvalue] == 0:
            continue
        else:
            break
    for maxvalue in range(255,minvalue,-1):
        if HistGram_1D[maxvalue] == 0:
            continue
        else:
            break
    if minvalue == maxvalue:
        return maxvalue
    if minvalue + 1 == maxvalue:
        return minvalue
    print(minvalue,maxvalue)
    amount = 0
    for i in range(minvalue,maxvalue+1):
        amount = amount + HistGram_1D[i]
    HistGramD = [0.0] * 256
    for i in range(minvalue,maxvalue+1):
        HistGramD[i] = float(HistGram_1D[i] / amount)
    maxEntropy = 0.0
    thresh = 0
    for i in range(minvalue+1,maxvalue):
        SumIntegral = 0.0
        EntropyBack = 0.0
        EntropyFore = 0.0
        for j in range(minvalue,i+1):
            SumIntegral = SumIntegral + HistGramD[j]
        for j in range(minvalue,i+1):
            if HistGramD[j] != 0:
                EntropyBack = EntropyBack + (-HistGramD[j] / SumIntegral * math.log(HistGramD[j] / SumIntegral))
        for j in range(i+1,maxvalue+1):
            if HistGramD[j] != 0:
                EntropyFore = EntropyFore + (-HistGramD[j] / (1 - SumIntegral) * math.log((HistGramD[j]) / (1 - SumIntegral)))
        if maxEntropy < (EntropyFore + EntropyBack):
            thresh = i
            maxEntropy = EntropyBack + EntropyFore
    return thresh

# Determine whether the histogram is a bimodal curve
# 判断直方图是否是双峰曲线
def IsDimodal(HistGram):
    Count = 0
    for Y in range(1, 255):
        if HistGram[Y - 1] < HistGram[Y] and HistGram[Y + 1] < HistGram[Y]:
            Count = Count + 1
            if Count > 2:
                return False
    if Count == 2:
        return True
    else:
        return False

# 2-Mode method
# 双峰阈值
def GetIntermodesThreshold(image):
    size = image.shape[0] * image.shape[1]
    HistGram = image.ravel()
    HistGram_1D = [0] * 256
    for i in range(size):
        index = HistGram[i]
        HistGram_1D[index] = HistGram_1D[index] + 1
    HistGramS = [float(0)] * 256
    HistGramC = [float(0)] * 256
    HistGramCC = [float(0)] * 256
    for i in range(256):
        HistGramC[i] = float(HistGram_1D[i])
        HistGramCC[i] = float(HistGram_1D[i])
    Iter = 0

    while IsDimodal(HistGramCC) == False:
        HistGramCC[0] = float((HistGramC[0] + HistGramC[0] + HistGramC[1]) / 3)
        for Y in range(1, 255):
            HistGramCC[Y] = float((HistGramC[Y - 1] + HistGramC[Y] + HistGramC[Y + 1]) / 3)
        HistGramCC[255] = float((HistGramC[254] + HistGramC[255] + HistGramC[255]) / 3)
        HistGramC = HistGramCC[:]
        Iter = Iter + 1
        if Iter >= 1000:
            return -1
    for Y in range(1, 256):
        HistGramS[Y] = int(HistGramCC[Y])
    Index = 0
    Peak = [None] * 2
    for Y in range(1, 255):
        if HistGramCC[Y - 1] < HistGramCC[Y] and HistGramCC[Y + 1] < HistGramCC[Y]:
            Peak[Index] = Y - 1
            Index = Index + 1
    binary_thresh = int((Peak[0] + Peak[1]) / 2)
    return binary_thresh

# Mean method
# 均值法二值化
def mean_threshold(image):
    height, width = image.shape[:2]
    m = np.reshape(image, [1,width * height])
    mean_binary = m.sum()/(width * height)
    return mean_binary