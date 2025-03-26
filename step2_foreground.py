import matplotlib.pyplot as plt
#import shapely.geometry
import cv2
import numpy as np
#import util_contours
import os
import traceback
#import facial_landmarks
import sys

def getForeground(img, mask_img):
    #height, width = mask_img.shape[:2]
    mask_img_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)

    (thresh, mask_img_bw) = cv2.threshold(mask_img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #mask_img_bw = cv2.threshold(mask_img_gray, thresh, 255, cv2.THRESH_BINARY)[1]
    
    #invert mask (mask for the background)
    mask_img_bw = cv2.bitwise_not(mask_img_bw)
    
    #to make background black
    img_background_black = cv2.bitwise_and(img, img, mask = mask_img_bw)
    
    #to make background transparent
    res = np.dstack((img_background_black, mask_img_bw))

    return res

def foreground_scene(SCENE_PATH):
    inputpathOriginal = SCENE_PATH+"/imagesFull"
    inputpath = SCENE_PATH+"/samtrack"
    outputpath = SCENE_PATH+"/out_foreground/"

    if not os.path.exists(outputpath):
       os.makedirs(outputpath)

    for filename in sorted(os.listdir(inputpath)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):# and filename=="00066.png":
            print("process("+inputpath+"/"+filename+", "+outputpath+")")
            
            mask_img = cv2.imread(os.path.join(inputpath, filename))
            assert mask_img is not None, "file could not be read, check with os.path.exists()"
           
            filename_without_extension= os.path.splitext(os.path.basename(filename))[0]
            img = cv2.imread(os.path.join(inputpathOriginal, filename_without_extension+".jpg"))
            assert img is not None, "file could not be read, check with os.path.exists()"
           
            foreground_img = getForeground(img, mask_img)

            cv2.imwrite(os.path.join(outputpath, filename), foreground_img)
            
if __name__ == "__main__":
    SCENE_PATH = sys.argv[1]

    foreground_scene(SCENE_PATH)                