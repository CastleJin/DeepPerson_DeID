import cv2
import numpy as np
import matplotlib.pyplot as plt
from vedacore.image import imread, imwrite
import glob
import os
from copy import deepcopy
from skimage.util import random_noise
from skimage.segmentation import slic
from skimage.measure import regionprops
from tqdm import tqdm
# 0002_c5s1_069073_02.jpg
# 0387_c3s1_091167_00.jpg

is_all = True
# goal)) change the face to anonymized face.
out_path = '/storage/sjhwang/target_all'
#root = '/storage/sjhwang/reid/market1501/market_o'
root = '/storage/sjhwang/TransReID-SSL/data'
imgs = '/*.jpg'
query = root + '/query'
train = root + '/bounding_box_train'
test = root + '/bounding_box_test'

q = glob.glob(query+imgs)
tr = glob.glob(train+imgs)
te = glob.glob(test+imgs)

stra = ['/L_blur/market1501', '/G_blur_sig5/market1501', '/Blank/market1501', '/Noise_sig30/market1501', '/Super/market1501', '/Edge/market1501']

def L_blur(img, f1=30, f2=60):
    out = deepcopy(img)
    h, w, c = out.shape
    for i in range(c):
        roi = cv2.resize(out[:,:,i], (f1, f2))
        roi = cv2.resize(roi, (w,h), interpolation=cv2.INTER_AREA)
        out[:,:,i] = roi
    return out

def G_blur(img, sigma=5): # 3
    out = deepcopy(img)
    out = cv2.GaussianBlur(out, (0, 0), sigma)
    return out

def Blank(img):
    out = deepcopy(img)
    out = np.zeros_like(out)
    return out

def Noise(img, mean=0, sig = 30):
    out = deepcopy(img)
    gauss = np.random.normal(mean,sig,(out.shape[0],out.shape[1]))
    noise = np.zeros_like(out)
    for i in range(3):
        noise[:,:,i] = out[:,:,i] + gauss
    return noise

def Super(img):
    out = deepcopy(img)
    def paint_region_with_avg_intensity(rp, mi, channel):
        for i in range(rp.shape[0]):
            out[rp[i][0]][rp[i][1]][channel] = mi
    segments = slic(out, n_segments=50, compactness=10,
                    channel_axis=2,
                    enforce_connectivity=True,
                    convert2lab=True)

    for i in range(3):
        regions = regionprops(segments, intensity_image=out[:,:,i])
        for r in regions:
            paint_region_with_avg_intensity(r.coords, int(r.mean_intensity), i)
    return out

def Edge(img):
    out = deepcopy(img)
    gray = deepcopy(out)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    ed = cv2.Canny(gray, 30, 70)
    for i in range(3):
        out[:,:,i] = ed     
    return out

def Filter(cropped, state):
    #print(state)
    if state == stra[0]:
        return L_blur(cropped)
    elif state == stra[1]:
        return G_blur(cropped)
    elif state == stra[2]:
        return Blank(cropped)
    elif state == stra[3]:
        return Noise(cropped)
    elif state == stra[4]:
        return Super(cropped)
    elif state == stra[5]:
        return Edge(cropped)
    else:
        print('Error. Check the current filter_state')

def set():
    for i in stra:
        j = out_path + i
        if not os.path.isdir(j):
            os.mkdir(j)
            os.mkdir(j + '/query')
            os.mkdir(j + '/bounding_box_test')
            os.mkdir(j + '/bounding_box_train')
            
def crop(img, x, y, w, h):
    cropped_img = img[y:y+h, x:x+w]
    return cropped_img

def replace(img, c, x, y, w, h):
    img[y:y+h, x:x+w] = c
    return img

def renew(boxes):
    le = []
    is_it = -1
    for i in boxes:
        le.append(len(i))
    for num, item in enumerate(le):
        if item == 1:
            is_it = num
            
    if is_it > 0:
        boxes[is_it-1].append(boxes[is_it][0])
        boxes.remove(boxes[is_it])
    return boxes

def reject(x,y,w,h,test):
    if w*h > 400:
        return True
    
    if x < 10 or (x+w) > 54:
        return True
    
    if h > 30:
        return True
    
    if not test:
        if y > 20 or y == 0:
            return True
        else:
            return False
    
    return False
    
        
def change(imgfp, test=''):
    img = imread(imgfp)
    text = imgfp.replace('jpg', 'txt')
    text = open(text, "r")
    boxes = []
    num = 0
    
    # Read
    while True:
        line = text.readline()
        if (not line): break
        box = line.split()
        tmp = [i.strip('['']') for i in box]
        for i in tmp:
            if i == '':
                tmp.remove('')
        new = [float(i) for i in tmp]
        boxes.append(new)
        num += 1
    text.close()
    
    if is_all:
        num = 0
    ori_pth = imgfp.replace(root, '')
    if num == 0:
        # Case 1. # of bbox = 0
        for i in stra:
            n_path = out_path + i + ori_pth
            c = deepcopy(img)
            re1 = Filter(c,i)
            imwrite(re1, n_path)  
    # 0505_c4s3_009248_00.jpg
    else:
        # Case 2. # of bbox > 0
        # save the (image and x, y, w, h)
        # img is cv2 style format.
        check = 0
        for i in range(3):
            boxes = renew(boxes)
        images = []
        
        for j, box in enumerate(boxes):
            box_int = box[:4]
            box_int = [round(i) for i in box_int]
            x = box_int[0]
            y = box_int[1]
            w = box_int[2]-box_int[0]
            h = box_int[3]-box_int[1]
            
            # choose the box
            if test == '':
                rr = reject(x,y,w,h,False)
            else:
                rr = reject(x,y,w,h,True)
            if rr and check==1:
                continue
            if is_all:
                rr = True
                check = 0
            for numm, i in enumerate(stra):
                if check == 0:
                    if rr:
                        # save img
                        n_path = out_path + i + ori_pth
                        c = deepcopy(img)
                        filter_result = Filter(c,i)
                        images.append(filter_result)
                        imwrite(filter_result, n_path)    
                    else:
                        if is_all:
                            c = deepcopy(img)
                        else:
                            c = crop(img, x, y, w, h)
                        # filter the cropped img.
                        filtered_c = Filter(c, i)
                        # replace the ROI.
                        replaced_img = deepcopy(img)
                        result = replace(replaced_img, filtered_c, x, y, w, h)
                        # save the img.
                        n_path = out_path + i + ori_pth
                        ### change the "filtered_c" -> result.
                        images.append(result)
                        # n_path
                        imwrite(result, n_path)
                else:
                    c = crop(images[numm], x, y, w, h)
                    # filter the cropped img.
                    filtered_c = Filter(c, i)
                    # replace the ROI.
                    replaced_img = deepcopy(images[numm])
                    result = replace(replaced_img, filtered_c, x, y, w, h)
                    # save the img.
                    n_path = out_path + i + ori_pth
                    ### change the "filtered_c" -> result.
                    images[numm] = result
                    imwrite(result, n_path)
            check = 1
        del images
                    
def main():
    set()
    for k in tqdm(te):
        change(k)
if __name__ == '__main__':
    main()


