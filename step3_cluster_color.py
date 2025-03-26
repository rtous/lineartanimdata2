import matplotlib.pyplot as plt
#import shapely.geometry
import cv2
import numpy as np
#import util_contours
import os
import traceback
#import facial_landmarks
import sys
#from sklearn.cluster import MeanShift, estimate_bandwidth

def removeBlack(img):
    color = (0,0,0)
    mask = np.where((img==color).all(axis=2), 0, 255).astype(np.uint8)

    # put mask into alpha channel
    result = img.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask
    return result

'''
def clustercolor_meanshift(img):
    # filter to reduce noise
    img = cv2.medianBlur(img, 3)

    # flatten the image
    flat_image = img.reshape((-1,3))
    flat_image = np.float32(flat_image)

    # meanshift
    bandwidth = estimate_bandwidth(flat_image, quantile=.06, n_samples=3000)
    ms = MeanShift()
    #ms = MeanShift(bandwidth, max_iter=800, bin_seeding=True)
    #ms = MeanShift(bandwidth)
    #ms = MeanShift(bandwidth, bin_seeding=True)
    #MeanShift(bandwidth=2).fit(X)
    #ms = MeanShift(1.5)
    ms.fit(flat_image)
    labeled=ms.labels_

    # get number of segments
    segments = np.unique(labeled)

    # get the average color of each segment
    total = np.zeros((segments.shape[0], 3), dtype=float)
    count = np.zeros(total.shape, dtype=float)
    for i, label in enumerate(labeled):
        total[label] = total[label] + flat_image[i]
        count[label] += 1
    avg = total/count
    avg = np.uint8(avg)

    # cast the labeled image into the corresponding average color
    res = avg[labeled]
    result = res.reshape((img.shape))

    return result
'''

def clustercolor_kmeans(img_orig, num_clusters):
    # resize
    #height, width = float(img_orig.shape[0]), float(img_orig.shape[1])
    #if width > height:
    #    new_width, new_height = (512, int(512 / width * height))
    #else:
    #    new_width, new_height = (int(512 / height * width), 512)
    #image = cv2.resize(img_orig, (new_width, new_height))
            
    rounds=1
    h, w = img_orig.shape[:2]
    samples = np.zeros([h*w,3], dtype=np.float32)
    #samples = np.zeros([h*w,4], dtype=np.float32)
    count = 0

    for x in range(h):
        for y in range(w):
            #if not (image[x][y][0] == 0 and image[x][y][1] == 0 and image[x][y][2] == 0):
            samples[count] = img_orig[x][y]
            count += 1

    compactness, labels, centers = cv2.kmeans(samples,num_clusters, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001), rounds, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    img_res = res.reshape((img_orig.shape))
    
    #resize to original
    #output = cv2.resize(img_res, (img_orig.shape[1], img_orig.shape[0]))
            
    return img_res

def clustercolor_scene(SCENE_PATH, num_clusters):
    inputpathForeground = SCENE_PATH+"/out_foreground"
    outputpath = SCENE_PATH+"/out_clustercolor/"

    if not os.path.exists(outputpath):
       os.makedirs(outputpath)

    for filename in sorted(os.listdir(inputpathForeground)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):# and filename=="00066.png":
            print("process("+inputpathForeground+"/"+filename+", "+outputpath+")")
            
            foreground_img = cv2.imread(os.path.join(inputpathForeground, filename), cv2.IMREAD_UNCHANGED)
            #foreground_img = cv2.imread(os.path.join(inputpathForeground, filename))
            
            #Apply a 3x3 gaussian kernel (average) to reduce noise
            kernel = np.ones((3,3),np.float32)/9
            foreground_img = cv2.filter2D(foreground_img,-1,kernel)
            #foreground_img = cv2.GaussianBlur(foreground_img, (0,0), sigmaX=33, sigmaY=33)

            b_channel, g_channel, r_channel, a_channel = cv2.split(foreground_img)
            foreground_img_noalpha = cv2.merge((b_channel, g_channel, r_channel))
            
            assert foreground_img is not None, "file could not be read, check with os.path.exists()"
           
            #colorReduce()
            #div = 256
            #clustered_img = foreground_img_noalpha // div * div + div // 2

            clustered_img = clustercolor_kmeans(foreground_img_noalpha, num_clusters)
            #clustered_img = clustercolor_meanshift(foreground_img_noalpha)

            b_channel, g_channel, r_channel = cv2.split(clustered_img)
            clustered_img = cv2.merge((b_channel, g_channel, r_channel, a_channel))

            cv2.imwrite(os.path.join(outputpath, filename), clustered_img)
            
if __name__ == "__main__":
    SCENE_PATH = sys.argv[1]
    if (len(sys.argv)>2):
        num_clusters = int(sys.argv[2])
    else:
        num_clusters = 10

    clustercolor_scene(SCENE_PATH, num_clusters)                