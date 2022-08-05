import os
import preprocessing
import segmenting
import time

'''
Deal with apple
'''

if __name__ == "__main__":
    # The first step, binarization
    start = time.time()
    picture_path = os.listdir(r"D:/CAPTCHA_Papers/Code/real-world/apple/image")
    for filelist in picture_path:
        open_path = "D:/CAPTCHA_Papers/Code/real-world/apple/image/" + filelist
        save_path = "D:/CAPTCHA_Papers/Code/real-world/apple/image_binary/" + filelist[:-4] + ".png"
        preprocessing.binary_apple(open_path, save_path)

    # The second step, corrosion, make the font outline thicker
    picture_path = os.listdir(r"D:/CAPTCHA_Papers/Code/real-world/apple/image_binary")
    for filelist in picture_path:
        open_path = "D:/CAPTCHA_Papers/Code/real-world/apple/image_binary/" + filelist
        save_path = "D:/CAPTCHA_Papers/Code/real-world/apple/image_erosion/" + filelist
        preprocessing.erosion_line(open_path, save_path)

    # The third step, segementing by vertical projection
    picture_path = os.listdir(r"D:/CAPTCHA_Papers/Code/real-world/apple/image_erosion/")
    for filelist in picture_path:
        open_path = "D:/CAPTCHA_Papers/Code/real-world/apple/image_erosion/" + filelist
        save_path = "D:/CAPTCHA_Papers/Code/real-world/apple/image_val_projection/" + filelist[:-4]
        character = filelist.split("_")[1].split(".")[0]
        segmenting.getVProjection(open_path, save_path, character,edge=2,extend=3)

    time = time.time() - start
    print("time", time)

    # The fourth step, horizontal projection to crop the upper and lower blank of the image
    picture_path = os.listdir(r"D:/CAPTCHA_Papers/Code/real-world/apple/image_val_projection/")
    for filelist in picture_path:
        open_path = "D:/CAPTCHA_Papers/Code/real-world/apple/image_val_projection/" + filelist
        save_path = "D:/CAPTCHA_Papers/Code/real-world/apple/image_val_segmentation/" + filelist[:-4]
        segmenting.getHProjection(open_path, save_path,edge=1)