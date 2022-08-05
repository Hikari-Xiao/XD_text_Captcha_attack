import cv2 as cv
import os
import numpy as np

"""
Segementation related Integration
"""

# Pay attention to these two parameters,
# edge：threshold about determining the image boundary;
# extend: The number of pixels extended of each character when splitting the image by vertical projection

# Segment by vertical projection
def getVProjection(open_path,save_path,character,edge,extend):
    Image = cv.imread(open_path,0)
    vProjection = np.zeros(Image.shape,np.uint8)
    (height,width) = Image.shape
    numofwhite = [0] * width
    for i in range(0,width):
        for j in range(0,height):
            if Image[j][i] == 255:
                numofwhite[i] = numofwhite[i] + 1
    print(numofwhite)
    for i in range(0,width):
        for j in range(0,numofwhite[i]):
            vProjection[j][i] = 255
    save_path_projection = save_path + "_projection.png"
    cv.imwrite(save_path_projection,vProjection)
    startIndex = 0
    endIndex = 0
    for i in range(1, width - 1):
        if (height - numofwhite[i]) >= edge  and (height - numofwhite[i - 1]) >= edge and (height - numofwhite[i + 1]) >= edge:
            startIndex = i
            break
    for i in range(width - 2, 0, -1):
        if (height - numofwhite[i]) >= edge and (height - numofwhite[i - 1]) >= edge and (height - numofwhite[i + 1]) >= edge:
            endIndex = i + 1
            break
    print(startIndex, endIndex)
    lenth = len(character)
    average_lenth = int((endIndex - startIndex) / lenth)
    for i in range(lenth):
        if i == 0:
            start = startIndex
        else:
            start = startIndex + average_lenth * i - extend
        if i < lenth - 1:
            end = startIndex + average_lenth * (i + 1) + extend
        else:
            end = endIndex
        cropped = Image[0:height, start:end]
        save_path1 = save_path + "_" + character[i] + ".png"
        cv.imwrite(save_path1, cropped)
    os.remove(save_path_projection)

# Segment by horizontal projection
def getHProjection(open_path,save_path,edge):
    Image = cv.imread(open_path,0)
    hProjection = np.zeros(Image.shape,np.uint8)
    (height,width) = Image.shape
    numofwhite = [0] * height
    for i in range(0,height):
        for j in range(0,width):
            if Image[i][j] == 255:
                numofwhite[i] = numofwhite[i] + 1
    print(numofwhite)
    for i in range(0,height):
        for j in range(0,numofwhite[i]):
            hProjection[i][j] = 255
    save_path_projection = save_path + "_hProjection.png"
    cv.imwrite(save_path_projection,hProjection)
    startIndex = 0
    endIndex = 0
    for i in range(1, height):
        if (width - numofwhite[i]) >= edge and (width - numofwhite[i - 1]) >= edge and (width - numofwhite[i + 1]) >= edge:
            startIndex = i
            break
        if i == height - 1:
            startIndex = 1
    for i in range(height-2, 0, -1):
        if (width - numofwhite[i]) >= edge and (width - numofwhite[i - 1]) >= edge and (width - numofwhite[i + 1]) >= edge:
            endIndex = i + 1
            break
        if i == 1:
            endIndex = height
    print(startIndex, endIndex)
    cropped = Image[startIndex:endIndex,0:width]
    save_path1 = save_path + ".png"
    cv.imwrite(save_path1, cropped)
    os.remove(save_path_projection)