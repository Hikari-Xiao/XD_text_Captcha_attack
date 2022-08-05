import os
import segmenting
import time

'''
Deal with wiki
'''
# The image of wiki is black and white and does not require special binarization processing

if __name__ == "__main__":
    # The first step, segementing by vertical projection
    start = time.time()
    picture_path = os.listdir(r"D:/CAPTCHA_Papers/Code/real-world/wiki/val")
    for filelist in picture_path:
        open_path = "D:/CAPTCHA_Papers/Code/real-world/wiki/val/" + filelist
        save_path = "D:/CAPTCHA_Papers/Code/real-world/wiki/wiki_projection_val/" + filelist[:-4]
        character = filelist.split("_")[1].split(".")[0]
        segmenting.getVProjection(open_path, save_path, character,edge=2,extend=3)

    time = time.time() - start
    print("time", time)

    # The second step, horizontal projection to crop the upper and lower blank of the image
    picture_path = os.listdir(r"D:/CAPTCHA_Papers/Code/real-world/wiki/wiki_projection_val")
    for filelist in picture_path:
        open_path = "D:/CAPTCHA_Papers/Code/real-world/wiki/wiki_projection_val/" + filelist
        save_path = "D:/CAPTCHA_Papers/Code/real-world/wiki/wiki_segmentation_val/" + filelist[:-4]
        segmenting.getHProjection(open_path, save_path,edge=1)