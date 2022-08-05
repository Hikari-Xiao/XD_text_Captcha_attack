#encoding:utf-8
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw, ImageFont, ImageFilter
import math
import os
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
zi_list = []
f = open("3000+class","r")
tmp_line = f.readline()
while tmp_line!="":
    zi_list.append(tmp_line.split(" ")[0].decode("utf-8"))
    tmp_line = f.readline()
for z in zi_list:
    print z.encode("utf-8")
Width = 100
Height = 40
adhere_left_len_max = 25#左偏移最大值
adhere_right_len_max = 25#右偏移最大值
adhere_left_height_max = 10#左图上下偏移最大值
adhere_right_height_max = 10#右图上下偏移最大值

warp_scale_x_max = 15#x方向的扭曲最大值
warp_scale_y_max = 15#y方向的扭曲最大值
warp_theta_x_max = math.pi/25.#x方向theta的变化最大速度
warp_theta_y_max = math.pi/25.#y方向theta的变化最大速度

reduce_len_max = 10

class img_shape_Exception(Exception):
    pass
def test_img_shape(img):
    if img.shape[0]<5 or img.shape[1]<5:
        raise img_shape_Exception()
def random_font():
    fonts = os.listdir("font/")
    index = np.random.randint(0,len(fonts))
    font_size = np.random.randint(20,35)
    return ImageFont.truetype('./font/'+fonts[index], font_size)

def random_zi():
    index = np.random.randint(0,len(zi_list))
    return zi_list[index],index

def dingge(image,key_list):
    left,upper,right,lower=500,500,-1,-1
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if (len(image.shape) ==2 and image[i,j]!=255) or (len(image.shape) ==3 and (image[i,j]!=[255 for m in range(image.shape[2])]).any()):
                if i<left:
                    left=i
                if i>right:
                    right=i
                if j<upper:
                    upper=j
                if j>lower:
                    lower=j
    res_list = []
    for k in key_list:
        if k[1]>=left and k[1]<right:
            if k[0]>=upper and k[0]<lower:
                res_list.append([k[0]-upper,k[1]-left])
    if len(image.shape) ==3:
        return image[left:right+1,upper:lower+1,:],res_list
    else:
        return image[left:right+1, upper:lower+1],res_list

def gen_image():
    font = random_font()
    zi,index = random_zi()
    img = Image.new("RGB",(45,45),(255,255,255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), zi, fill=(0, 0, 0), font=font)
    return np.asarray(img),index

def adhere(ori_img,key_list):
    left_img,_ = gen_image()
    left_img,_ = dingge(left_img,[])
    left_img, _ = rotate(left_img, [])

    right_img,_ = gen_image()
    right_img,_ = dingge(right_img,[])
    right_img, _ = rotate(right_img, [])

    left_img_len = adhere_left_len_max
    right_img_len = adhere_right_len_max
    ad_left_len = 0
    if left_img_len!=0:
        ad_left_len = np.random.randint(0,10)
    ad_right_len = 0
    if right_img_len!=0:
        ad_right_len = np.random.randint(0,10)
    ad_left_height = np.random.randint(0,adhere_left_height_max)
    ad_right_height = np.random.randint(0,adhere_right_height_max)

    res_img_width = ori_img.shape[1]+left_img_len+right_img_len-ad_left_len-ad_right_len
    res_img_height = ori_img.shape[0]
    if len(ori_img.shape) == 2:
        res_img = np.full((res_img_height,res_img_width),255,dtype=np.uint8)
    else:
        res_img = np.full((res_img_height, res_img_width,ori_img.shape[2]), 255, dtype=np.uint8)
    start_i = 0
    i = 0
    for i in range(start_i,left_img_len):
        for j in range(res_img_height):
            ori_i = left_img.shape[1]-left_img_len+i
            ori_j = j+ad_left_height
            if ori_i>=0 and ori_i<left_img.shape[1] and ori_j>=0 and ori_j<left_img.shape[0]:
                if len(res_img.shape)==2 and left_img[ori_j,ori_i]!=255:
                    res_img[j,i] = left_img[ori_j,ori_i]
                if len(res_img.shape)==3 and (left_img[ori_j,ori_i]!=[255,255,255]).any():
                    res_img[j,i] = left_img[ori_j,ori_i]
    start_i = i-ad_left_len
    for i in range(start_i,start_i+ori_img.shape[1]):
        for j in range(res_img_height):
            ori_i = i-start_i
            ori_j = j
            if ori_i>=0 and ori_i<ori_img.shape[1] and ori_j>=0 and ori_j<res_img_height:
                if len(res_img.shape)==2 and ori_img[ori_j,ori_i]!=255:
                    res_img[j,i] = ori_img[ori_j,ori_i]
                if len(res_img.shape)==3 and (ori_img[ori_j,ori_i]!=[255,255,255]).any():
                    res_img[j,i] = ori_img[ori_j,ori_i]
    start_i = i - ad_right_len
    for i in range(start_i, res_img_width):
        for j in range(res_img_height):
            ori_i = i - start_i
            ori_j = j+ad_right_height
            if ori_i >= 0 and ori_i < right_img.shape[1] and ori_j >= 0 and ori_j < right_img.shape[0]:
                if len(res_img.shape) == 2 and res_img[j,i]==255:
                    res_img[j, i] = right_img[ori_j, ori_i]
                if len(res_img.shape) == 3 and (res_img[j,i]==[255,255,255]).all():
                    res_img[j, i] = right_img[ori_j, ori_i]
    ori_pos_left = [left_img_len-ad_left_len,res_img_height]
    ori_pos_right = [left_img_len+ori_img.shape[1]-ad_left_len,0]
    for i in range(len(key_list)):
        key_list[i][0] += left_img_len-ad_left_len
    return res_img, key_list

def warp(ori_img,key_list):
    fuc_width = ori_img.shape[1]
    fuc_height = ori_img.shape[0]
    scale_x = np.random.randint(0,warp_scale_x_max)
    scale_y = np.random.randint(0,warp_scale_y_max)
    x_theta = warp_theta_x_max*2*(np.random.random_sample()-0.5)
    y_theta = warp_theta_y_max*2*(np.random.random_sample()-0.5)
    x_theta_start = 2 * 2 * math.pi * (np.random.random_sample() - 0.5)
    y_theta_start = 2 * 2 * math.pi * (np.random.random_sample() - 0.5)
    res_img = None
    if len(ori_img.shape) == 2:
        res_img = np.full((fuc_height+2*scale_y,fuc_width+2*scale_x),255,dtype=np.uint8)
    else:
        res_img = np.full((fuc_height + 2 * scale_y, fuc_width + 2 * scale_x, ori_img.shape[2]), 255, dtype=np.uint8)
    res_list = []

    for i in range(res_img.shape[1]):
        for j in range(res_img.shape[0]):
            ori_i = int(i - scale_x * math.sin(x_theta * j+x_theta_start) - scale_x)
            # ori_i = int(i-scale_x)
            ori_j = int(j - scale_y * math.sin(y_theta * i+y_theta_start) - scale_y)
            for k in key_list:
                if ori_i == k[0] and ori_j == k[1]:
                    res_list.append([i,j])
            # ori_j = int(j - scale_y)
            # if ori_i == ori_pos_left[0] and ori_j == ori_pos_left[1]:
            #     ori_pos_left[0] = i
            #     ori_pos_left[1] = j
            # if ori_i == ori_pos_right[0] and ori_j == ori_pos_right[1]:
            #     ori_pos_right[0] = i
            #     ori_pos_right[1] = j

            if ori_i>=0 and ori_i<ori_img.shape[1] and ori_j>=0 and ori_j<ori_img.shape[0]:
                res_img[j,i] = ori_img[ori_j,ori_i]
    return res_img,res_list

def rotate(ori_img,key_list):
    theta = math.pi/4.*(np.random.random_sample()-0.5)*2
    if len(ori_img.shape) == 2:
        res_img = np.full((2*ori_img.shape[1],2*ori_img.shape[0]),255,dtype=np.uint8)
    else:
        res_img = np.full((2*ori_img.shape[1],2*ori_img.shape[0], ori_img.shape[2]), 255, dtype=np.uint8)
    res_width = res_img.shape[1]
    res_height = res_img.shape[0]
    res_list = []
    for i in range(res_width):
        for j in range(res_height):
            p_i = i - res_width/3.
            p_j = j - res_height/3.
            ori_i = int(p_i*math.cos(-theta)-p_j*math.sin(-theta))
            ori_j = int(p_j*math.cos(-theta)+p_i*math.sin(-theta))
            for k in key_list:
                if ori_i == k[0] and ori_j == k[1]:
                    res_list.append([i,j])
            if ori_i>=0 and ori_i<ori_img.shape[1] and ori_j>=0 and ori_j<ori_img.shape[0]:
                res_img[j,i] = ori_img[ori_j,ori_i]
    return dingge(res_img,res_list)

def noise_line(ori_img):
    line_sum = np.random.randint(3,6)
    img_H = ori_img.shape[0]
    img_W = ori_img.shape[1]
    d_list = [[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1],[-1,0],[-1,1]]
    for l in range(line_sum):
        l_pos = [np.random.randint(0,img_H),np.random.randint(0,img_W)]
        l_pos_real = [np.random.randint(0,img_H),np.random.randint(0,img_W)]
        l_v  = d_list[np.random.randint(0,8)]
        l_a = [np.random.rand(),np.random.rand()]
        l_sum = np.random.randint(20,100)
        for s in range(l_sum):
            ori_img[l_pos[0],l_pos[1]] = [0, 0, 0]
            l_pos_real[0] = l_pos_real[0] + l_v[0]
            if l_pos_real[0]>=40:
                l_pos_real[0] = 39
            if l_pos_real[0]<0:
                l_pos_real[0] = 0
            l_pos_real[1] = l_pos_real[1] + l_v[1]
            if l_pos_real[1]>=100:
                l_pos_real[1] = 99
            if l_pos_real[0]<0:
                l_pos_real[0] = 0

            l_pos[0] = int(l_pos_real[0])
            l_pos[1] = int(l_pos_real[1])
            l_a_rate = np.random.rand()
            if l_a_rate > 0.9:
                l_v[0] += l_a[0]
                l_v[1] += l_a[1]
                if l_v[0]<l_v[1]:
                    l_v[0]/=l_v[1]
                    l_v[1]/=l_v[1]
                else:
                    l_v[0] /= l_v[0]
                    l_v[1] /= l_v[0]
            l_a_rate = np.random.rand()
            if l_a_rate >1:
                l_a = [np.random.rand(), np.random.rand()]
    return ori_img

def reduce_img(image):
    rate = np.random.rand()
    l = 0
    if rate > 0.5:
        l = np.random.randint(0,reduce_len_max)
    rate = np.random.rand()
    r = 0
    if rate > 0.5:
        r = np.random.randint(0,reduce_len_max)
    return image[:,l:image.shape[1]-r]

def resize_key_list(res,key_list,width,height):
    res_list = []
    res_width_rate = (width+0.)/res.shape[1]
    res_height_rate = (height+0.)/res.shape[0]
    for k in key_list:
        res_list.append([int(k[0]*res_width_rate),
                         int(k[1]*res_height_rate)])
    return res_list

def dingge_key(image,key_list):
    left, upper, right, lower = 500, 500, -1, -1
    for k in key_list:
        if k[1] < left:
            left = k[1]
        if k[1] > right:
            right = k[1]
        if k[0] < upper:
            upper = k[0]
        if k[0] > lower:
            lower = k[0]
    if len(image.shape) ==3:
        return image[left:right+1,upper:lower+1,:]
    else:
        return image[left:right+1 , upper:lower+1]

def draw_key(image,key_list):
    res = image
    for k in key_list:
        res[k[1],k[0]] = [0,0,255]
    return res

def draw_img(image):
    draw_color = [1,0,0]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if (image[i, j] != [255,255,255]).any():
                image[i, j] = draw_color
                draw_color = next_color(draw_color)
    return image

def eliminate_img(res,label_img):
    color_list = []
    max_value = 0
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            if (res[i][j] != [255,255,255]).any():
                tmp_color = [res[i][j][0],res[i][j][1],res[i][j][2]]
                color_list.append(tmp_color)
                res[i,j] = [0,0,0]
                tmp_value = tmp_color[0]+tmp_color[1]*256+tmp_color[2]*256*256
                if tmp_value > max_value:
                    max_value = tmp_value
                # threshold = 155#二值化阙值
                # tmp_v = (res[i,j,0]+res[i,j,1]+res[i,j,2])/3
                # if tmp_v > threshold:
                #     res[i][j] = [255,255,255]
                # else:
                #     res[i][j] = [0, 0, 0]
    color_count_list = np.full(max_value+1,0)

    for c in color_list:
        color_count_list[c[0]+c[1]*256+c[2]*256*256] = 1

    for i in range(label_img.shape[0]):
        for j in range(label_img.shape[1]):
            tmp_value = label_img[i,j,0]+256*label_img[i,j,1]+256*256*label_img[i,j,2]
            if tmp_value<=max_value\
                    and color_count_list[tmp_value]==1:
                label_img[i,j] = [0,0,0]
            else:
                label_img[i, j] = [255,255,255]

def random_resize(img):
    img_height = img.shape[0]
    img_width = img.shape[1]
    ran_width = np.random.randint(15,25)
    ran_height = int((img_height+0.)/img_width*ran_width)
    img = cv2.resize(img,(ran_width,ran_height))
    return img

def color_value(color):
    res = float(color[0])+color[1]+color[2]
    return res
def gen_ran_res_img():
    first_img,first_index = gen_image()
    first_img,_ = dingge(first_img,[])
    second_img,second_index = gen_image()
    second_img,_ = dingge(second_img, [])
    try:
        test_img_shape(first_img)
        test_img_shape(second_img)
    except img_shape_Exception:
        return None,None,None
    first_img = random_resize(first_img)
    second_img = random_resize(second_img)
    try:
        test_img_shape(first_img)
        test_img_shape(second_img)
    except img_shape_Exception:
        return None,None,None
    first_img,_ = rotate(first_img,[])
    second_img,_ = rotate(second_img,[])
    try:
        test_img_shape(first_img)
        test_img_shape(second_img)
    except img_shape_Exception:
        return None,None,None
    res_img = np.full((40,100,3),255,np.uint8)
    first_img_H = first_img.shape[0]
    first_img_W = first_img.shape[1]
    second_img_H = second_img.shape[0]
    second_img_W = second_img.shape[1]
    if 40 - first_img_H <= 0 :
        first_img_pos_H = 0
    else:
        first_img_pos_H = np.random.randint(0, 40 - first_img_H)
    first_img_pos_W = np.random.randint(0, 100 - first_img_W-second_img_W-1)
    adhere_len = np.random.randint(-5,5)
    if 40 - second_img_H <= 0 :
        second_img_pos_H = 0
    else:
        second_img_pos_H = np.random.randint(0, 40 - second_img_H)
    second_img_pos_W = first_img_pos_W+first_img_W + adhere_len
    for i in range(100):
        for j in range(40):
            if i-first_img_pos_W >=0 and i-first_img_pos_W < first_img_W:
                if j-first_img_pos_H >=0 and j-first_img_pos_H < first_img_H:
                    if color_value(res_img[j,i]) > color_value(first_img[j-first_img_pos_H,i-first_img_pos_W]):
                        res_img[j, i] = first_img[j-first_img_pos_H,i-first_img_pos_W]
            if i-second_img_pos_W >=0 and i-second_img_pos_W < second_img_W:
                if j-second_img_pos_H >=0 and j-second_img_pos_H < second_img_H:
                    if color_value(res_img[j,i]) > color_value(second_img[j-second_img_pos_H,i-second_img_pos_W]):
                        res_img[j, i] = second_img[j-second_img_pos_H,i-second_img_pos_W]
    res_img = noise_line(res_img)
    ran_t = np.random.randint(150,180)
    _,res_img = cv2.threshold(res_img,ran_t,255,cv2.THRESH_BINARY)
    return res_img,first_index,second_index

def next_color(ori_color):
    p0 = ori_color[0]
    p1 = ori_color[1]
    p2 = ori_color[2]
    p0 += 1
    if p0 > 255:
        p0 = 0
        p1 += 1
        if p1 > 255:
            p1 = 0
            p2 += 1
            if p2 >255:
                print "too many color!"
                p2 = 0
    return [p0, p1, p2]
f_ans = open("test.txt","w")
for i in range(1000):
    print i
    res,f,s = gen_ran_res_img()
    if res is None:
        i-=1
        continue
    cv2.imwrite("test_imgs/{0}.png".format(i),res)
    f_ans.write("{0}.png {1} {2}\n".format(i,f,s))

# f = open("train.txt","w")
# for i in range(848000):
#     print "{0}/848000".format(i)
#     res,index = gen_ran_res_img()
#     if res == None:
#         continue
#     cv2.imwrite("./img/{0}_{1}.png".format(index,i),res)
#     f.write("{0}_{1}.png {0}\n".format(index,i))
# f.close()
    

# tst = draw_key(res,key_list)
# cv2.imshow("test",tst)
# res = dingge_key(res,key_list)
# cv2.imshow("res",res)
