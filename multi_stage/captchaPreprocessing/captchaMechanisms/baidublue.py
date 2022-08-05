import os
import preprocessing
import segmenting
import time

'''
Deal with baidu_blue
baidublue验证码处理
'''

if __name__ == "__main__":
    # The first step, binarization
    start = time.time()
    picture_path = os.listdir(r"D:/CAPTCHA_Papers/Code/real-world/baiduBlue/val")
    for filelist in picture_path:
        open_path = "D:/CAPTCHA_Papers/Code/real-world/baiduBlue/val/" + filelist
        save_path = "D:/CAPTCHA_Papers/Code/real-world/baiduBlue/val_binary/" + filelist[:-4]
        preprocessing.binary(open_path, save_path,"fixed_threshold")

    # The second step, segementing by vertical projection
    picture_path = os.listdir(r"D:/CAPTCHA_Papers/Code/real-world/baiduBlue/val_binary")
    for filelist in picture_path:
        open_path = "D:/CAPTCHA_Papers/Code/real-world/baiduBlue/val_binary/" + filelist
        save_path = "D:/CAPTCHA_Papers/Code/real-world/baiduBlue/val_projection/" + filelist[:-4]
        character = filelist.split("_")[1].split(".")[0]
        segmenting.getVProjection(open_path, save_path, character,edge=2,extend=3)

    time = time.time() - start
    print("time", time)

    # The third step, horizontal projection to crop the upper and lower blank of the image
    picture_path = os.listdir(r"D:/CAPTCHA_Papers/Code/real-world/baiduBlue/val_projection")
    for filelist in picture_path:
        open_path = "D:/CAPTCHA_Papers/Code/real-world/baiduBlue/val_projection/" + filelist
        save_path = "D:/CAPTCHA_Papers/Code/real-world/baiduBlue/val_segmentation/" + filelist[:-4]
        segmenting.getHProjection(open_path, save_path,edge=1)