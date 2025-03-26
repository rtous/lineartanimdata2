import matplotlib.pyplot as plt
#import shapely.geometry
import cv2
import numpy as np
#import util_contours
import os
import traceback
#import facial_landmarks
import sys

def removeBlack(img):
    color = (0,0,0)
    mask = np.where((img==color).all(axis=2), 0, 255).astype(np.uint8)

    # put mask into alpha channel
    result = img.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask
    return result

def edges(image):
    clusters=10
    rounds=1
    h, w = image.shape[:2]
    samples = np.zeros([h*w,3], dtype=np.float32)
    #samples = np.zeros([h*w,4], dtype=np.float32)
    count = 0

    for x in range(h):
        for y in range(w):
            #if not (image[x][y][0] == 0 and image[x][y][1] == 0 and image[x][y][2] == 0):
            samples[count] = image[x][y]
            count += 1

    compactness, labels, centers = cv2.kmeans(samples,clusters, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001), rounds, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape))

    return res

def increase_contrast(img):
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img

def edges_scene(SCENE_PATH):
    inputpath = SCENE_PATH+"/out_clustercolor"
    outputpath = SCENE_PATH+"/out_sketch_canny_from_clustercolor/"

    if not os.path.exists(outputpath):
       os.makedirs(outputpath)

    for filename in sorted(os.listdir(inputpath)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):# and filename=="00066.png":
            print("process("+inputpath+"/"+filename+", "+outputpath+")")
            
            img = cv2.imread(os.path.join(inputpath, filename), cv2.IMREAD_UNCHANGED)
            #alpha to white
            trans_mask = img[:,:,3] == 0
            #replace areas of transparency with white and not transparent
            img[trans_mask] = [255, 255, 255, 255]
            
            #increase contrast
            img = increase_contrast(img)


            img_orig_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #edge_img = img_orig_grayscale
            edge_img = cv2.Canny(img_orig_grayscale, 0, 255)
            edge_img = cv2.bitwise_not(edge_img)

            cv2.imwrite(os.path.join(outputpath, filename), edge_img)
            
if __name__ == "__main__":
    SCENE_PATH = sys.argv[1]

    edges_scene(SCENE_PATH)                