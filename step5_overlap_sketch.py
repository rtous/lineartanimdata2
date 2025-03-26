import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import traceback
import sys

def removeBlackBackground(img):
    alpha = np.sum(img, axis=-1) > 0
    # Convert True/False to 0/255 and change type to "uint8" to match "na"
    alpha = np.uint8(alpha * 255)
    img = np.dstack((img, alpha))

def overlay(bottomImage, topImage):
    #Idea: add the topImage (complete) to a sliced bottomImage 
    #Obtain an opencvmask from the alpha channel of the topImage
    _, mask = cv2.threshold(topImage[:, :, 3], 0, 255, cv2.THRESH_BINARY)
    #Invert the mask
    mask = cv2.bitwise_not(mask) 
    #Use the mask to cut the intersection from the bottomImage
    bottomImageMinusTopImage = cv2.bitwise_and(bottomImage, bottomImage, mask=mask)
    #Add the topImage (complete) and bottomImageMinusTopImage
    result = bottomImageMinusTopImage + topImage
    return result


def opencv_to_RGB(c):
    return c[::-1]

def drawContours(contours, imcolor):
    for i, contour in enumerate(contours):
        cv2.drawContours(imcolor, [contour], contourIdx=0, color=(0,0,0), thickness=2)        
    return imcolor

def cropContours(im, contour):
    im_res = np.zeros_like(im)
    cv2.fillPoly(im_res, pts =[contour], color=(255,255,255))
    return im_res

def getContours(im):
    height, width = im.shape[:2]
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    contours_not_dilated = []
    contours_raw = []
    contours_simplified = []
    colors = []

    #split image in C color regions (with a minimum of 1000 pixels)
    selected_contours = []
    contours_simplified = [] 
    colorNum = 0
    totalContours = 0
    #unique = np.unique(imgray)
    unique_colours = np.unique(im.reshape(-1, im.shape[2]), axis=0)
    #For each COLOR 
    for i, color in enumerate(unique_colours):
        mask = cv2.inRange(im, color, color)
        area = cv2.countNonZero(mask)
        if area > 200 and area < height*width/2: #avoid the frame contour
            #split color mask in N contours (with a minimum of area > 10)
            ret, thresh = cv2.threshold(mask, 127, 255, 0)
            image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for j, contour in enumerate(contours):
                if cv2.contourArea(contour) > 10:
                    contours_not_dilated.append(contour)
                    print("Color (opencv)="+str(colorNum)+"="+str(color))
                    #dilate 1 pixel (to avoid gaps between simplified contours)
                    #remove anything outside the contour
                    part_mask = cropContours(mask, contour)
                    kernel = np.ones((4, 4), np.uint8)
                    part_mask = cv2.dilate(part_mask, kernel, iterations=1)
                    #cv2.imshow("title", part_mask)
                    #cv2.waitKey() 

                    #find contours again
                    ret, thresh = cv2.threshold(part_mask, 127, 255, 0)
                    image, contours_dilated, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                    max_contour = max(contours_dilated, key = cv2.contourArea)

                    contours_raw.append(max_contour)
                    print("colors.append(opencv_to_RGB("+str(color)+"))")
                    colors.append(opencv_to_RGB(color))
                    test = np.zeros_like(imgray)
                    #cv2.drawContours(test, [max_contour], contourIdx=0, color=(100,200,100), thickness=2)
                    #cv2.imshow("title", test)
                    #cv2.waitKey() 
                    totalContours = totalContours+1                          
            colorNum = colorNum+1
    print("Found "+str(colorNum)+" colors")
    print("Found "+str(totalContours)+" contours")
    return contours_not_dilated, contours_raw, colors

def getLineart(img):
    
    contours_not_dilated, contours_raw, colors = getContours(img)
    #cv2.drawContours(imcolor, [contour], contourIdx=0, color=display_color, thickness=1)        
    drawContours(contours_not_dilated, img)
    return img

def lineart_scene(SCENE_PATH, from_cluster):
    inputpath = SCENE_PATH+"/out_sketch_fromclustercolor"
    if (from_cluster == 1):
        inputpathColor = SCENE_PATH+"/out_clustercolor"
    else:
        inputpathColor = SCENE_PATH+"/out_foreground"
    outputpath = SCENE_PATH+"/out_overlap_sketch/"
    outputpathsketchalpha = SCENE_PATH+"/out_overlap_sketchalpha/"

    if not os.path.exists(outputpath):
       os.makedirs(outputpath)

    if not os.path.exists(outputpathsketchalpha):
       os.makedirs(outputpathsketchalpha)

    for filename in sorted(os.listdir(inputpath)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):# and filename=="00066.png":
            print("process("+inputpath+"/"+filename+", "+outputpath+")")
            
            img_sketch = cv2.imread(os.path.join(inputpath, filename))
            assert img_sketch is not None, "file could not be read, check with os.path.exists()"
            
            filename_without_extension= os.path.splitext(os.path.basename(filename))[0]
            img_color = cv2.imread(os.path.join(inputpathColor, filename_without_extension+".png"), cv2.IMREAD_UNCHANGED)
            assert img_color is not None, "file could not be read, check with os.path.exists()"
           
            img_sketch_gray = cv2.cvtColor(img_sketch, cv2.COLOR_BGR2GRAY)
            (thresh, img_sketch_bw) = cv2.threshold(img_sketch_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            #mask_img_bw = cv2.threshold(mask_img_gray, thresh, 255, cv2.THRESH_BINARY)[1]
            
            #invert mask (mask for the background)
            mask_img_sketch_bw = cv2.bitwise_not(img_sketch_bw)
            
            #to make background black
            img_sketch_background_black = cv2.bitwise_and(img_sketch, img_sketch, mask = mask_img_sketch_bw)
            
            #to make background transparent
            img_sketch_alpha = np.dstack((img_sketch_background_black, mask_img_sketch_bw))

            #cv2.imwrite(os.path.join(outputpath, filename), img_sketch_alpha)
            #cv2.imwrite(os.path.join(outputpath, filename), img_color)

            #alpha_channel = np.zeros(b_channel.shape, dtype=b_channel.dtype)
            #res = overlay(img_color, img_sketch_alpha)
            
            #b_channel, g_channel, r_channel = cv2.split(img_color)
            #alpha_channel = np.zeros(b_channel.shape, dtype=b_channel.dtype)
            #img_color = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
            

            #make sketch black
            img_sketch_alpha = cv2.threshold(img_sketch_alpha, thresh, 255, cv2.THRESH_BINARY)[1]
    

            #img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2BGRA)

            res = img_color + img_sketch_alpha
            #res = img_color
            #res = img_sketch_alpha


            

            res = overlay(img_color, img_sketch_alpha)

            cv2.imwrite(os.path.join(outputpathsketchalpha, filename), img_sketch_alpha)

            cv2.imwrite(os.path.join(outputpath, filename), res)
            
if __name__ == "__main__":
    SCENE_PATH = sys.argv[1]
    if (len(sys.argv)>2):
        from_cluster = int(sys.argv[2]) #0=original, 1=from clustered colors
    else:
        from_cluster = 1

    lineart_scene(SCENE_PATH, from_cluster)                