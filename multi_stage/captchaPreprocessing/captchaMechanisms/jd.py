import os
import preprocessing
import segmenting
import time

'''
Deal with JD
'''

if __name__ == "__main__":
    # The first step, binarization
    start = time.time()
    picture_path = os.listdir(r"D:/CAPTCHA_Papers/Code/real-world/jd/image")
    for filelist in picture_path:
        open_path = "D:/CAPTCHA_Papers/Code/real-world/jd/image/" + filelist
        save_path = "D:/CAPTCHA_Papers/Code/real-world/jd/image_binary1/" + filelist[:-4]
        preprocessing.binary_jd(open_path, save_path)

    # The second step, blacken the outline of the character
    picture_path = os.listdir(r"D:/CAPTCHA_Papers/Code/real-world/jd/image_binary1")
    for filelist in picture_path:
        open_path = "D:/CAPTCHA_Papers/Code/real-world/jd/image_binary1/" + filelist
        save_path = "D:/CAPTCHA_Papers/Code/real-world/jd/image_convert1/" + filelist
        preprocessing.erosion_line(open_path, save_path)

    # The third step, segementing by vertical projection
    picture_path = os.listdir(r"D:/CAPTCHA_Papers/Code/real-world/jd/image_convert1")
    for filelist in picture_path:
        open_path = "D:/CAPTCHA_Papers/Code/real-world/jd/image_convert1/" + filelist
        save_path = "D:/CAPTCHA_Papers/Code/real-world/jd/image_projection1/" + filelist[:-4]
        character = filelist.split("_")[1].split(".")[0]
        segmenting.getVProjection(open_path, save_path, character,edge=2,extend=3)

    time = time.time() - start
    print("time", time)

    # The fourth step, horizontal projection to crop the upper and lower blank of the image
    picture_path = os.listdir(r"D:/CAPTCHA_Papers/Code/real-world/jd/image_projection1")
    for filelist in picture_path:
        open_path = "D:/CAPTCHA_Papers/Code/real-world/jd/image_projection1/" + filelist
        save_path = "D:/CAPTCHA_Papers/Code/real-world/jd/image_segmentation1/" + filelist[:-4]
        segmenting.getHProjection(open_path, save_path,edge=1)