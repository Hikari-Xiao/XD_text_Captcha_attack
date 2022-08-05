import os
import preprocessing
import segmenting
import time

'''
Deal with 360 gray
'''

if __name__ == "__main__":
    # The first step, remove the backgrounds

    start = time.time()
    picture_path = os.listdir(r"D:/CAPTCHA_Papers/Code/real-world/360gray/image")
    for filelist in picture_path:
        open_path = "D:/CAPTCHA_Papers/Code/real-world/360gray/image/" + filelist
        save_path = "D:/CAPTCHA_Papers/Code/real-world/360gray/image_removebackground/" + filelist[:-4] + ".png"
        preprocessing.remove_backgroud_360gray(open_path,save_path)

    # The second step, binarization
    picture_path = os.listdir(r"D:/CAPTCHA_Papers/Code/real-world/360gray/image_removebackground")
    for filelist in picture_path:
        open_path = "D:/CAPTCHA_Papers/Code/real-world/360gray/image_removebackground/" + filelist
        save_path = "D:/CAPTCHA_Papers/Code/real-world/360gray/image_binary/" + filelist[:-4]
        preprocessing.binary(open_path,save_path,"Iterative_best_threshold")

    # The third step, corrosion and bold
    picture_path = os.listdir(r"D:/CAPTCHA_Papers/Code/real-world/360gray/image_binary")
    for filelist in picture_path:
        open_path = "D:/CAPTCHA_Papers/Code/real-world/360gray/image_binary/" + filelist
        save_path = "D:/CAPTCHA_Papers/Code/real-world/360gray/image_erosion/" + filelist
        preprocessing.erosion_line(open_path, save_path)

    # The fourth step, segementing by vertical projection
    picture_path = os.listdir(r"D:/CAPTCHA_Papers/Code/real-world/360gray/image_erosion")
    for filelist in picture_path:
        open_path = "D:/CAPTCHA_Papers/Code/real-world/360gray/image_erosion/" + filelist
        save_path = "D:/CAPTCHA_Papers/Code/real-world/360gray/image_projection/" + filelist[:-4]
        character = filelist.split("_")[1].split(".")[0]
        segmenting.getVProjection(open_path, save_path, character, edge = 2, extend = 3)

    time = time.time() - start
    print("time:",time)

    # The fifth step, horizontal projection to crop the upper and lower blank of the image
    picture_path = os.listdir(r"D:/CAPTCHA_Papers/Code/real-world/360gray/image_projection")
    for filelist in picture_path:
        open_path = "D:/CAPTCHA_Papers/Code/real-world/360gray/image_projection/" + filelist
        save_path = "D:/CAPTCHA_Papers/Code/real-world/360gray/image_segmentation/" + filelist[:-4]
        segmenting.getHProjection(open_path, save_path, edge = 1)