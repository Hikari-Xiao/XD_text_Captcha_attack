import os
import cv2 as cv
import preprocessing
import segmenting
import time

'''
Deal with baidu
baidu验证码处理
'''

if __name__ == '__main__':
    # The first step, crop the image to remove the background image of the bear
    # 第一步，裁剪图像去掉小熊背景图像
    start = time.time()
    picture = os.listdir(r"D:/CAPTCHA_Papers/Code/real-world/baidu/baidu_image")
    for filelist in picture:
        img = cv.imread("D:/CAPTCHA_Papers/Code/real-world/baidu/baidu_image/" + filelist)
        cropped = img[0:27,0:100]
        save_path = "D:/CAPTCHA_Papers/Code/real-world/baidu/baidu_cropped/" + filelist[:-4] + ".png"
        cv.imwrite(save_path,cropped)

    # The second step, binarization
    # 第二步，二值化
    picture_path = os.listdir(r"D:/CAPTCHA_Papers/Code/real-world/baidu/baidu_cropped")
    for filelist in picture_path:
        open_path = "D:/CAPTCHA_Papers/Code/real-world/baidu/baidu_cropped/" + filelist
        save_path = "D:/CAPTCHA_Papers/Code/real-world/baidu/baidu_binary/" + filelist[:-4]
        # preprocessing.binary(open_path, save_path, "MaxEntropy_1D")
        preprocessing.binary(open_path, save_path, "mean_threshold")

    # The third step, remove the noise lines
    # 第三步，去除噪线
    # 膨胀去除噪线
    picture_path = os.listdir(r"D:/CAPTCHA_Papers/Code/real-world/baidu/baidu_binary")
    for filelist in picture_path:
        open_path = "D:/CAPTCHA_Papers/Code/real-world/baidu/baidu_binary/" + filelist
        save_path = "D:/CAPTCHA_Papers/Code/real-world/baidu/baidu_dilation_result/" + filelist
        preprocessing.dilation(open_path, save_path)

    # The fourth step, segementing by vertical projection
    # 第四步，垂直投影分割
    picture_path = os.listdir(r"D:/CAPTCHA_Papers/Code/real-world/baidu/baidu_dilation_result")
    for filelist in picture_path:
        open_path = "D:/CAPTCHA_Papers/Code/real-world/baidu/baidu_dilation_result/" + filelist
        save_path = "D:/CAPTCHA_Papers/Code/real-world/baidu/baidu_projection/" + filelist[:-4]
        character = filelist.split("_")[1].split(".")[0]
        segmenting.getVProjection(open_path,save_path,character,edge=2,extend=2)

    time = time.time() - start
    print("time", time)

    # The fifth step, horizontal projection to crop the upper and lower blank of the image
    # 第五步，水平投影实现顶格
    picture_path = os.listdir(r"D:/CAPTCHA_Papers/Code/real-world/baidu/baidu_projection")
    for filelist in picture_path:
        open_path = "D:/CAPTCHA_Papers/Code/real-world/baidu/baidu_projection/" + filelist
        save_path = "D:/CAPTCHA_Papers/Code/real-world/baidu/baidu_segment/" + filelist[:-4]
        segmenting.getHProjection(open_path,save_path,edge=2)
