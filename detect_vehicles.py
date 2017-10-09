import numpy as np
import cv2
import glob
import imageio
from moviepy.editor import VideoFileClip
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
# from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.metrics import accuracy_score
import pickle
from scipy.ndimage.measurements import label

# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy
    
def convert_color(img, color_space='YCrCb'):
    if color_space == 'RGB':
        return np.copy(img)
    if color_space == 'YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if color_space == 'HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if color_space == 'LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if color_space == 'HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if color_space == 'YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, color_space, orient, pix_per_cell, cell_per_block):
    img = img.astype(np.float32)/255

    bboxes = []
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, color_space=color_space)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step_x = 2  # Instead of overlap, define how many cells to step
    cells_per_step_y = 2  # Instead of overlap, define how many cells to step
    nxsteps = (ch1.shape[1] - window) // (cells_per_step_x * pix_per_cell) + 1
    nysteps = (ch1.shape[0] - window) // (cells_per_step_y * pix_per_cell) + 1
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step_y
            xpos = xb*cells_per_step_x
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            test_prediction = svc.predict(hog_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                bboxes.append(((xbox_left, ytop_draw+ystart), (xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                
    return bboxes

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def label_to_bboxes(labels):
    bboxes = []
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        bboxes.append(bbox)
    # Return the image
    return bboxes

def detect(image):
    detect.count += 1
    bbox_list = []
    for scale in [1.0, 1.5, 2.0, 2.5, 3.0]:
        ystart = 400
        ystop = min(720, int(ystart + scale * 64 * 1.5))
        bboxes = find_cars(image, ystart, ystop, scale, detect.clf, detect.color_space, detect.orient, detect.pix_per_cell, detect.cell_per_block)
        bbox_list.extend(bboxes)

    # draw_image = draw_boxes(image, bbox_list)
    # mpimg.imsave('./output_images/single_detect_' + str(detect.count) + ".png", draw_image)
    # return draw_image

    detect.last_bbox_list.append(bbox_list)
    if len(detect.last_bbox_list) > 10:
        detect.last_bbox_list.pop(0)

    # heat = np.zeros_like(image[:,:,0]).astype(np.float)
    # heat = add_heat(heat, bbox_list)
    # mpimg.imsave('./output_images/single_heat_' + str(detect.count) + ".png", heat)    

    # Add heat to each box in box list
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    for bbox_list in detect.last_bbox_list:
        heat = add_heat(heat, bbox_list)
    # mpimg.imsave('./output_images/combined_heat_' + str(detect.count) + ".png", heat)    
        
    # Apply threshold to help remove false positives
    if len(detect.last_bbox_list) == 0:
        heat = apply_threshold(heat, 1)
    else:
        heat = apply_threshold(heat, 4)
    # mpimg.imsave('./output_images/threshold_heat_' + str(detect.count) + ".png", heat)    

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    # mpimg.imsave('./output_images/label_heat_' + str(detect.count) + ".png", labels[0])    

    bboxes = label_to_bboxes(labels)

    draw_image = draw_boxes(image, bboxes)
    # mpimg.imsave('./output_images/combined_detect_' + str(detect.count) + ".png", draw_image)
    return draw_image

detect.clf = pickle.load(open('svc_model.pickle', 'rb'))
detect.color_space = "YCrCb" # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
detect.orient = 9
detect.pix_per_cell = 8
detect.cell_per_block = 2
detect.hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"

images = glob.glob('./test_images/test*.jpg')
for image_name in images:
    print(image_name)
    image = mpimg.imread(image_name)
    detect.count = 0
    detect.last_bbox_list = []
    draw_image = detect(image)
    mpimg.imsave('./output_images/sliding_windows_result_' + image_name[14:], draw_image)

def process_image(image):
    return detect(image)
    
detect.last_bbox_list = []
detect.count = 0
input_video_filename = "project_video.mp4"
output_video_filename = "project_video_output.mp4"
input_video = VideoFileClip(input_video_filename).subclip(30, 32)
output_video = input_video.fl_image(process_image)
output_video.write_videofile(output_video_filename, audio=False)
