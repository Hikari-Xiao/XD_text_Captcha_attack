import cv2 as cv
import numpy as np
import thresholdCaculating

"""
Preprocess Integration
Integrate various pretreatment methods together
Including binarization, corrosion, expansion, rotation and other methods 
"""

# input: the image in the open_path
# output: save the processed image in the save_path

# Binarization
# threMethod: the binarization method you choose
def binary(open_path, save_path, threMethod, fixedThre=220, gray = "False"):
    image = cv.imread(open_path, 0)
    # Choose one binarization method
    if(threMethod == "GetPTileThreshold"):
        # threshold = thresholdCaculating.GetPTileThreshold(image)
        threshold = thresholdCaculating.GetPTileThreshold(open_path, gray = gray)
    elif(threMethod == "average_threshold"):
        img = cv.imread(open_path)
        image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        threshold = thresholdCaculating.average_threshold(image)
    elif (threMethod == "Iterative_best_threshold"):
        image = cv.imread(open_path)
        threshold = thresholdCaculating.Iterative_best_threshold(image, gray = gray)
    elif (threMethod == "MaxEntropy_1D"):
        threshold = thresholdCaculating.average_threshold(open_path)
    elif (threMethod == "GetIntermodesThreshold"):
        threshold = thresholdCaculating.average_threshold(image)
    elif (threMethod == "mean_threshold"):
        threshold = thresholdCaculating.mean_threshold(image)
    # Binarization by using certain threshold
    elif(threMethod == "fixed_threshold"):
        threshold = fixedThre
    ret, binary_result = cv.threshold(image, threshold, 255, cv.THRESH_BINARY)
    save_path_binary = save_path + ".png"
    # print(save_path_binary)
    # print(binary_result)
    cv.imwrite(save_path_binary, binary_result)

# related to Rotation
# Detect the upper left corner of the image to determine whether
# the image needs to be rotated clockwise or counterclockwise
def clockwise_or_anticlockwise(open_path):
    image = cv.imread(open_path,0)
    sum = 0
    for i in range(30):
        for j in range(100):
            if image[i][j] == 0:
                sum = sum + 1
    if sum > 1:
        print(open_path + "逆时针")
        return -1
    else:
        print(open_path + "顺时针")
        return 1

# Rotation
def rotate_bound(open_path,save_path,angle):
    image = cv.imread(open_path,0)
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)
    angle = angle * clockwise_or_anticlockwise(open_path)
    M = cv.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    img = cv.warpAffine(image, M, (nW, nH),borderValue=(255,255,255))
    cv.imwrite(save_path,img)

# Expansion
def dilation(open_path,save_path):
    image = cv.imread(open_path,0)
    kernel = np.ones((2, 2), np.uint8)
    dilation = cv.dilate(image, kernel)
    cv.imwrite(save_path, dilation)

# Corrosion and Bold
def erosion_line(open_path,save_path):
    image = cv.imread(open_path,0)
    kernel = np.ones((2, 2), np.uint8)
    erosion = cv.erode(image, kernel)
    cv.imwrite(save_path, erosion)

# Get the noise lines
def get_noiseline(open_path,save_path,first,k1size,k2size):
    image = cv.imread(open_path,0)
    kernel1 = np.ones((k1size, k1size), np.uint8)
    kernel2 = np.ones((k2size, k2size), np.uint8)
    # First dilate and remove characters to extract noise lines,
    # and then corrode to restore noise lines
    if first=="dilation":
        dilation = cv.dilate(image, kernel1)
        erosion = cv.erode(dilation, kernel2)
        cv.imwrite(save_path, erosion)
    # Expansion after corrosion
    else:
        erosion = cv.erode(image, kernel1)
        dilation = cv.dilate(erosion, kernel2)
        cv.imwrite(save_path, dilation)

# Remove the noise lines according to the original image and the obtained noise line
def remove_noiseline(open_path1,open_path2,save_path):
    image1 = cv.imread(open_path1,0)
    image2 = cv.imread(open_path2,0)
    (height,width) = image1.shape
    for i in range(height):
        for j in range(width):
            if image1[i][j] != 255:
                image1[i][j] = 0
            if image2[i][j] == 0:
                image1[i][j] = 255
    cv.imwrite(save_path,image1)

# Remove the noise lines and binarization
# 360_hollow
def remove_lines(path1,path2,path3):
    # path1: captcha image
    # path2: noise lines
    # path3: save_path
    img1 = cv.imread(path1)
    img2 = cv.imread(path2)
    (height,width,channel) = img2.shape
    for i in range(height):
        for j in range(width):
            for k in range(channel):
                if img1[i][j][k] == 0:
                    img2[i][j][k] = 255
    Img = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    ret, Img2 = cv.threshold(Img, 220, 255, cv.THRESH_BINARY)
    cv.imwrite(path3, Img2)

# Remove background(against 360_gray)
# three kinds of different backgrounds,
# which are backgrounds composed of straight lines, diagonal lines, and wavy lines.
# Need to extract the background template, remove the background by doing the difference
def remove_backgroud_360gray(open_path,save_path):
    image = cv.imread(open_path,0)
    # Get three kinds of background images
    Img1 = cv.imread("./360gray_background/break_line.jpg",0)
    Img2 = cv.imread("./360gray_background/oblique_line.jpg",0)
    Img3 = cv.imread("./360gray_background/transverse_line.jpg",0)
    img= image[2:10, 2:10]
    img1 = Img1[2:10, 2:10]
    img2 = Img2[2:10, 2:10]
    img3 = Img3[2:10, 2:10]
    flag1 = 0
    flag2 = 0
    flag3 = 0
    for i in range(0,8):
        for j in range(0,8):
            if abs(int(img[i][j])-int(img1[i][j])) <= 30:
                flag1 = flag1 + 1
            if abs(int(img[i][j]) - int(img2[i][j])) <= 30:
                flag2 = flag2 + 1
            if abs(int(img[i][j])-int(img3[i][j])) <= 30:
                flag3 = flag3 + 1
    (height, width) = image.shape
    Image = image[3:height - 3, 3:width - 3]
    h,w = Image.shape[:2]
    if flag1 > flag2 and flag1 > flag3:
        Image1 = Img1[3:height - 3, 3:width - 3]
        for i in range(h):
            for j in range(w):
                if abs(int(Image[i][j]) - int(Image1[i][j])) <= 30:
                    Image[i][j] = 255
        cv.imwrite(save_path, Image)

    elif flag2 > flag1 and flag2 > flag3:
        Image2 = Img2[3:height - 3, 3:width - 3]
        for i in range(h):
            for j in range(w):
                if abs(int(Image[i][j]) - int(Image2[i][j])) <= 30:
                    Image[i][j] = 255
        cv.imwrite(save_path, Image)
    elif flag3 > flag1 and flag3 > flag2:
        Image3 = Img3[3:height - 3, 3:width - 3]
        for i in range(h):
            for j in range(w):
                if abs(int(Image[i][j]) - int(Image3[i][j])) <= 30:
                    Image[i][j] = 255
        cv.imwrite(save_path, Image)

# Pseudo-binarization(against JD)
# Extract white characters and convert them to black
def binary_jd(open_path,save_path):
    image = cv.imread(open_path,0)
    (height, width) = image.shape
    img = np.zeros((height, width), np.uint8)
    for i in range(height):
        for j in range(width):
            img[i][j] = 255
    for i in range(height):
        for j in range(width):
            if image[i][j] == 255:
                img[i][j] = 0
    save_path1 = save_path + ".png"
    cv.imwrite(save_path1 ,img)

# Pseudo-binarization(against apple)
# Extract black characters
def binary_apple(open_path,save_path):
    image = cv.imread(open_path,0)
    (height,width) = image.shape
    for i in range(height):
        for j in range(width):
            if image[i][j] >= 7:
                image[i][j] = 255
    cv.imwrite(save_path,image)