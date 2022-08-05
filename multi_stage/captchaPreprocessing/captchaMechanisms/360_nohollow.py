import os
import preprocessing
import segmenting
import time

'''
Deal with 360 nohollow
'''


if __name__ == "__main__":
    # The first step, binarization
    start = time.time()
    picturepath = os.listdir(r"D:/CAPTCHA_Papers/Code/real-world/360_nohollow_pro/360_nohollow_image")
    for filelist in picturepath:
        open_path = "D:/CAPTCHA_Papers/Code/real-world/360_nohollow_pro/360_nohollow_image/" + filelist
        save_path = "D:/CAPTCHA_Papers/Code/real-world/360_nohollow_pro/360_nohollow_binary/"+ filelist[:-4]
        preprocessing.binary(open_path, save_path, "average_threshold")

    # The second step ，expanding to remove noise lines
    picturepath = os.listdir(r"D:/CAPTCHA_Papers/Code/real-world/360_nohollow_pro/360_nohollow_binary")
    for filelist in picturepath:
        open_path = "D:/CAPTCHA_Papers/Code/real-world/360_nohollow_pro/360_nohollow_binary/" + filelist
        save_path = "D:/CAPTCHA_Papers/Code/real-world/360_nohollow_pro/360_nohollow_removelines_result/"+ filelist[:-4] +  ".png"
        preprocessing.dilation(open_path, save_path)

    # The third step, segementing by vertical projection
    picturepath = os.listdir(r"D:/CAPTCHA_Papers/Code/real-world/360_nohollow_pro/360_nohollow_removelines_result")
    for filelist in picturepath:
        openpath = "D:/CAPTCHA_Papers/Code/real-world/360_nohollow_pro/360_nohollow_removelines_result/" + filelist
        save_path = "D:/CAPTCHA_Papers/Code/real-world/360_nohollow_pro/360_nohollow_segment/" + filelist[:-4]
        character = filelist.split("_")[1].split(".")[0]
        segmenting.getVProjection(openpath, save_path, character, edge = 3, extend = 2)
    time = time.time() - start
    print("time:",time)

    # The fourth step, horizontal projection to crop the upper and lower blank of the image
    picturepath = os.listdir(r"D:/CAPTCHA_Papers/Code/real-world/360_nohollow_pro/360_nohollow_segment")
    for filelist in picturepath:
        open_path = "D:/CAPTCHA_Papers/Code/real-world/360_nohollow_pro/360_nohollow_segment/" + filelist
        save_path = "D:/CAPTCHA_Papers/Code/real-world/360_nohollow_pro/360_nohollow_segment_result/" + filelist[:-4]
        segmenting.getHProjection(open_path, save_path, edge = 2)