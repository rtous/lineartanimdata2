import sys
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
from sam2.build_sam import build_sam2
#from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 
import cv2

def RGB_to_opencv(c):
    return c[::-1]

def main(scene_path):
    #COLOR PALETTE (from segment.py from lester-code)
    np.random.seed(200)
    _palette = ((np.random.random((3*255))*0.7+0.3)*255).astype(np.uint8).tolist()
    _palette = [0,0,0]+_palette
    #c = _palette[id*3:id*3+3] USAGE

    #Build our own dictionary for the colors
    palette = {}
    for i in range(255):
        palette[i] = _palette[i*3:i*3+3]


    #load model
    device = torch.device("mps")
    from sam2.build_sam import build_sam2_video_predictor
    sam2_checkpoint = "sam2/checkpoints/sam2.1_hiera_large.pt"
    #changed from the original:
    #model_cfg = "sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    #load frames
    # `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
    video_dir = scene_path+"/imagesFull"
    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    #initialize inference state
    inference_state = predictor.init_state(video_path=video_dir)

    '''
    #This takes the specific scene segmentation settings from the file data/scenes/test/scene_segmentation_settings.py
    import sys
    sys.path.append("data/scenes/test/")
    from scene_segmentation_settings import *
    '''

    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = 0  # give a unique id to each object we interact with (it can be any integers)

    for mask_points in masks_points:
        print("add_new_points_or_box for ann_obj_id="+str(ann_obj_id))
        print(mask_points[0])
        points = mask_points[0]
        labels = mask_points[1]
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )
        
        # show the results on the current (interacted) frame
        print("showing one mask")
        mask = (out_mask_logits[ann_obj_id] > 0.0).cpu().numpy()
        color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
        for point in points:
            plt.plot(point[0], point[1], marker="o", markersize=10, markeredgecolor="green", markerfacecolor="green")
        #plt.plot(x2, y2, marker="o", markersize=10, markeredgecolor="red", markerfacecolor="red")
        #plt.plot(x3, y3, marker="o", markersize=10, markeredgecolor="green", markerfacecolor="green")
        plt.gca().imshow(mask_image)
        plt.show()
        

        ann_obj_id = ann_obj_id+1;


    #PROPAGATION STAGE

    output_path = scene_path+"/samtrack"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #obtain dimensions of the image
    first_frame = cv2.imread(os.path.join(video_dir, frame_names[0]))
    height, width, channels = first_frame.shape

    #Propagate the prompts to get the masklet across the video
    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        print("frame done")
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        blank_image_BGR = np.zeros((height,width,3), np.uint8)
        blank_image_BGR[:] = 255
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            print("showing out_frame_idx="+str(out_frame_idx)+"/out_obj_id="+str(out_obj_id))
            mask = out_mask
            color = np.array([30/255, 144/255, 255/255, 0.6])
            h, w = mask.shape[-2:]
            mask = mask.astype(np.uint8)
            mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            '''
            fig = plt.figure(figsize=(10, 10))
            plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
            #plt.plot(x, y, marker="o", markersize=10, markeredgecolor="green", markerfacecolor="green")
            #plt.plot(x2, y2, marker="o", markersize=10, markeredgecolor="red", markerfacecolor="red")
            #plt.plot(x3, y3, marker="o", markersize=10, markeredgecolor="green", markerfacecolor="green")
            plt.gca().imshow(mask_image)
            plt.show()
            '''

            #mask is a vector of 0s and 1s. 
            #reshaping it as a matrix becomes a mask useful in opencv
            binary_mask = mask.reshape(h,w)
            
            #create a white blank image (3 channels)
            #blank_image_BGR = np.zeros((h,w,3), np.uint8)
            #blank_image_BGR[:] = 255

            #apply a color using the mask
            blank_image_BGR[binary_mask==1]=RGB_to_opencv(palette[(out_obj_id+10*1)%256])

        #save
        #cv2.imwrite(output_path+"/f"+str(out_frame_idx)+".png", blank_image_BGR)
        frame_filename_no_extension = os.path.splitext(frame_names[out_frame_idx])[0]
        cv2.imwrite(output_path+"/"+frame_filename_no_extension+'.png', blank_image_BGR)


    '''
    #IF YOU NEED TO DO IT LATER
    # show the results 
    print("showing last mask")
    for out_frame_idx in range(0, len(frame_names)):
        #create a white blank image (3 channels)
        blank_image_BGR = np.zeros((height,width,3), np.uint8)
        blank_image_BGR[:] = 255
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            print("showing out_frame_idx="+str(out_frame_idx)+"/out_obj_id="+str(out_obj_id))
            mask = out_mask
            color = np.array([30/255, 144/255, 255/255, 0.6])
            h, w = mask.shape[-2:]
            mask = mask.astype(np.uint8)
            mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        

            #mask is a vector of 0s and 1s. 
            #reshaping it as a matrix becomes a mask useful in opencv
            binary_mask = mask.reshape(h,w)
            
            #create a white blank image (3 channels)
            #blank_image_BGR = np.zeros((h,w,3), np.uint8)
            #blank_image_BGR[:] = 255

            #apply a color using the mask
            blank_image_BGR[binary_mask==1]=RGB_to_opencv(palette[(out_obj_id+10*1)%256])

        #save
        #cv2.imwrite(output_path+"/f"+str(out_frame_idx)+".png", blank_image_BGR)
        frame_filename_no_extension = os.path.splitext(frame_names[out_frame_idx])[0]
        cv2.imwrite(output_path+"/"+frame_filename_no_extension+'.png', blank_image_BGR)
    '''
    
if __name__ == "__main__":
    print("Ruben's SAM-Track launcher")
    scene_path = sys.argv[1]
    print("scene_path:", scene_path)

    #This takes the specific scene segmentation settings from the file data/scenes/test/scene_segmentation_settings.py
    sys.path.append(scene_path+"/")
    from scene_segmentation_settings import *

    main(scene_path)