import os
import preprocessing
import segmenting
import time

'''
Deal with 360 hollow
'''

if __name__ == "__main__":

    # The first step, binarization
    start = time.time()
    hollowpath = os.listdir(r"D:/CAPTCHA_Papers/Code/real-world/360_hollow_pro/360_hollow_image")
    for filelist in hollowpath:
        open_path = "D:/CAPTCHA_Papers/Code/real-world/360_hollow_pro/360_hollow_image/" + filelist
        save_path = "D:/CAPTCHA_Papers/Code/real-world/360_hollow_pro/360_hollow_binary/"+ filelist[:-4]
        preprocessing.binary(open_path, save_path, "GetPTileThreshold", gray = "True")

    # The second step, removing the noise lines
    # Corrosion and expansion
    binarypath = os.listdir(r"D:/CAPTCHA_Papers/Code/real-world/360_hollow_pro/360_hollow_binary")
    for filelist in binarypath:
        open_path = "D:/CAPTCHA_Papers/Code/real-world/360_hollow_pro/360_hollow_binary/" + filelist
        path1 = "D:/CAPTCHA_Papers/Code/real-world/360_hollow_pro/360_hollow_remove_lines/" + filelist
        preprocessing.get_noiseline(open_path, path1,first="erosion",k1size=3,k2size=3)
        path2 = "D:/CAPTCHA_Papers/Code/real-world/360_hollow_pro/360_hollow_image/" + filelist[:-4] + ".jpg"
        path3 = "D:/CAPTCHA_Papers/Code/real-world/360_hollow_pro/360_hollow_removelines_result/" + filelist
        preprocessing.remove_lines(path1,path2,path3)

    # The third step, segementing by vertical projection
    picturepath = os.listdir(r"D:/CAPTCHA_Papers/Code/real-world/360_hollow_pro/360_hollow_removelines_result")
    for filelist in picturepath:
        open_path = "D:/CAPTCHA_Papers/Code/real-world/360_hollow_pro/360_hollow_removelines_result/" + filelist
        save_path = "D:/CAPTCHA_Papers/Code/real-world/360_hollow_pro/360_hollow_segment/" + filelist[:-4]
        character = filelist.split("_")[1].split(".")[0]
        segmenting.getVProjection(open_path, save_path, character,edge=2,extend=2)
    end = time.time()
    time = end - start
    print("time",time)

    # The fourth step, horizontal projection to crop the upper and lower blank of the image
    picturepath = os.listdir(r"D:/CAPTCHA_Papers/Code/real-world/360_hollow_pro/360_hollow_segment")
    for filelist in picturepath:
        open_path = "D:/CAPTCHA_Papers/Code/real-world/360_hollow_pro/360_hollow_segment/" + filelist
        save_path = "D:/CAPTCHA_Papers/Code/real-world/360_hollow_pro/360_hollow_segment_result/" + filelist[:-4]
        segmenting.getHProjection(open_path, save_path,edge=2)