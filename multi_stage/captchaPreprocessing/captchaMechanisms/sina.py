import os
import preprocessing
import segmenting
import time

'''
Deal with sina
'''
# The background is white, the characters are blue or red, the characters of each captcha are the same color,
# there is only one noise line, and the color of the noise line is the same as the character

if __name__ == "__main__":
    # The first step, binarization
    start = time.time()
    picture_path = os.listdir(r"D:/CAPTCHA_Papers/Code/real-world/sina/image")
    for filelist in picture_path:
        open_path = "D:/CAPTCHA_Papers/Code/real-world/sina/image/" + filelist
        save_path = "D:/CAPTCHA_Papers/Code/real-world/sina/image_binary/" + filelist[:-4]
        preprocessing.binary(open_path, save_path, "GetPTileThreshold")

    # The second step, get the noise lines
    picture_path = os.listdir(r"D:/CAPTCHA_Papers/Code/real-world/sina/image_binary")
    for filelist in picture_path:
        open_path = "D:/CAPTCHA_Papers/Code/real-world/sina/image_binary/" + filelist
        save_path = "D:/CAPTCHA_Papers/Code/real-world/sina/image_noiseline/" + filelist[:-4] + ".png"
        preprocessing.get_noiseline(open_path, save_path,first="dilation",k1size=2,k2size=3)

    # The third step, remove the noise lines
    picture_path = os.listdir(r"D:/CAPTCHA_Papers/Code/real-world/sina/image")
    for filelist in picture_path:
        open_path1 = "D:/CAPTCHA_Papers/Code/real-world/sina/image/" + filelist
        open_path2 = "D:/CAPTCHA_Papers/Code/real-world/sina/image_noiseline/" + filelist[:-4] + ".png"
        save_path = "D:/CAPTCHA_Papers/Code/real-world/sina/image_remove_noiseline/" + filelist[:-4] + ".png"
        preprocessing.remove_noiseline(open_path1, open_path2, save_path)

    # The fourth step, segementing by vertical projection
    # 第四步，垂直投影分割
    picture_path = os.listdir(r"D:/CAPTCHA_Papers/Code/real-world/sina/image_remove_noiseline")
    for filelist in picture_path:
        open_path = "D:/CAPTCHA_Papers/Code/real-world/sina/image_remove_noiseline/" + filelist
        save_path = "D:/CAPTCHA_Papers/Code/real-world/sina/image_projection/" + filelist[:-4]
        character = filelist.split("_")[1].split(".")[0]
        segmenting.getVProjection(open_path, save_path, character,edge=2,extend=3)

    time = time.time() - start
    print("time", time)

    # The fifth step, horizontal projection to crop the upper and lower blank of the image
    picture_path = os.listdir(r"D:/CAPTCHA_Papers/Code/real-world/sina/image_projection")
    for filelist in picture_path:
        open_path = "D:/CAPTCHA_Papers/Code/real-world/sina/image_projection/" + filelist
        save_path = "D:/CAPTCHA_Papers/Code/real-world/sina/image_segmentation/" + filelist[:-4]
        segmenting.getHProjection(open_path, save_path,edge=1)