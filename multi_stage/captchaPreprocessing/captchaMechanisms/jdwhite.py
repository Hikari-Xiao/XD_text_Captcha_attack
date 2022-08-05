import os
import preprocessing
import segmenting
import time

'''
Deal with JD_white
'''

if __name__ == "__main__":
    # The first step, binarization
    start = time.time()
    picture_path = os.listdir(r"D:/CAPTCHA_Papers/Code/real-world/jdWhite/image")
    for filelist in picture_path:
        open_path = "D:/CAPTCHA_Papers/Code/real-world/jdWhite/image/" + filelist
        save_path = "D:/CAPTCHA_Papers/Code/real-world/jdWhite/image_binary/" + filelist[:-4]
        preprocessing.binary(open_path, save_path,"GetPTileThreshold")

    # The second step, segementing by vertical projection
    picture_path = os.listdir(r"D:/CAPTCHA_Papers/Code/real-world/jdWhite/image_binary")
    for filelist in picture_path:
        open_path = "D:/CAPTCHA_Papers/Code/real-world/jdWhite/image_binary/" + filelist
        save_path = "D:/CAPTCHA_Papers/Code/real-world/jdWhite/image_projection1/" + filelist[:-4]
        character = filelist.split("_")[1].split(".")[0]
        segmenting.getVProjection(open_path, save_path, character,edge=2,extend=3)

    time = time.time() - start
    print("time", time)

    # The third step, horizontal projection to crop the upper and lower blank of the image
    picture_path = os.listdir(r"D:/CAPTCHA_Papers/Code/real-world/jdWhite/image_projection1")
    for filelist in picture_path:
        open_path = "D:/CAPTCHA_Papers/Code/real-world/jdWhite/image_projection1/" + filelist
        save_path = "D:/CAPTCHA_Papers/Code/real-world/jdWhite/image_segmentation1/" + filelist[:-4]
        segmenting.getHProjection(open_path, save_path,edge=1)