import os
import preprocessing
import segmenting
import time

'''
Deal with microsoft
'''

if __name__ == "__main__":
    # The first step, binarization
    start = time.time()
    picturepath = os.listdir(r"D:/CAPTCHA_Papers/Code/real-world/ms/image")
    for filelist in picturepath:
        open_path = "D:/CAPTCHA_Papers/Code/real-world/ms/image/" + filelist
        save_path = "D:/CAPTCHA_Papers/Code/real-world/ms/image_binary/" + filelist[:-4]
        preprocessing.binary(open_path, save_path, "fixed_threshold", fixedThre=245)

    # The second step is to automatically rotate the image so that the characters are in a horizontal position
    picturepath = os.listdir(r"D:/CAPTCHA_Papers/Code/real-world/ms/image_binary")
    for filelist in picturepath:
        open_path = "D:/CAPTCHA_Papers/Code/real-world/ms/image_binary/" + filelist
        save_path = "D:/CAPTCHA_Papers/Code/real-world/ms/image_rotate/" + filelist
        preprocessing.rotate_bound(open_path, save_path, 45)

    # The third step, segementing by vertical projection
    picture_path = os.listdir(r"D:/CAPTCHA_Papers/Code/real-world/ms/image_rotate")
    for filelist in picture_path:
        open_path = "D:/CAPTCHA_Papers/Code/real-world/ms/image_rotate/" + filelist
        save_path = "D:/CAPTCHA_Papers/Code/real-world/ms/image_projection/" + filelist[:-4]
        character = filelist.split("_")[1].split(".")[0]
        segmenting.getVProjection(open_path, save_path, character,edge=2,extend=3)

    time = time.time() - start
    print("time", time)

    # The fourth step, horizontal projection to crop the upper and lower blank of the image
    picture_path = os.listdir(r"D:/CAPTCHA_Papers/Code/real-world/ms/image_projection")
    for filelist in picture_path:
        open_path = "D:/CAPTCHA_Papers/Code/real-world/ms/image_projection/" + filelist
        save_path = "D:/CAPTCHA_Papers/Code/real-world/ms/image_segmentation/" + filelist[:-4]
        segmenting.getHProjection(open_path, save_path,edge=1)