import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import imutils
np.set_printoptions(threshold=np.nan)

data_dir = 'cropped/teeth/volume/'
watershedImgs = os.listdir(data_dir)
labelList = []
watershedImageList = []
distanceCheckedList = []
center_coords = []

binary_slices_for_model = []

def show_image(image, second_image):
    cv2.imshow('image1', image)
    cv2.imshow('image2', second_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def find_centroid(image, idx):
    temp_center_coords = []
    height, width = np.shape(image)
    image = np.uint8(image)
    print(width, height)
    print(image[int(width/2), int(height/2)])

    #Make image binary, hold former centroids
    ret, centroidImage = cv2.threshold(image,1,1,cv2.THRESH_BINARY)
    print('thresholded image')
    contours = cv2.findContours(centroidImage.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if imutils.is_cv2() else contours[1]
    print('len contours: ', len(contours))

    #Put centroids in image
    colorCount = 2
    for num, c in enumerate(contours):
        #if num == len(contours)-1:
        #    break
        M = cv2.moments(c)
        cX = int(M['m10'] / (M['m00']+1)) #+1 to prevent 0 division
        cY = int(M['m01'] / (M['m00']+1)) #+1 to prevent 0 division
        print('coordinates: ', cX, cY)
        height1, width1 = np.shape(image)
        print('middle picture: ', width1/2, height1/2)
        cv2.drawContours(centroidImage, [c], -1, (0, 255, 0), 2)
        centers = cv2.circle(centroidImage, (cX, cY), 3, (colorCount, colorCount, colorCount), -1)
        colorCount += 2
        # Append cX and cY to global center_coords
        temp_center_coords.append((cX, cY))

    if len(contours) > 1:
        for x in range(0, height):
            for y in range(0, width):
                if image[x,y] > 1:
                    image[x,y] = 0
                if centers[x,y] > 1:
                    image[x,y] = centers[x,y]
    center_coords.append(temp_center_coords)
    '''
    print('show center image')
    plt.imshow(image, cmap='jet')
    plt.show()
    '''
    return image

def most_common_element(image):
    image = image.flatten()
    print(np.shape(image))
    for num, idx in enumerate(image):
        if idx < 0:
            image[num] = 1000 #Arbitrary number, to filter out negative numbers
    counts = np.bincount(image)
    return np.argmax(counts)

def findLabel(markers, width, height):
    deviation = 10
    middleX = int(width/2)
    middleY = int(height/2)

    center_image = markers[middleX-deviation : middleX+deviation, middleY-deviation : middleY+deviation]
    #Find the label of the center tooth
    label = most_common_element(center_image)
    print(label)
    return label

def distance_check(image, idx):
    centers = center_coords[idx]
    label = labelList[idx]
    height, width = np.shape(image)
    print(height, width)
    for x in range(height):
        for y in range(width):
            
            if (image[x,y] != 1) and (image[x,y] != -1):# or (image[x,y] == label):
                smallest_distance = 1000 #Set very large
                X = Y = 0
                for coords in centers:
                    cY = coords[0]
                    cX = coords[1]
                    #print(cX, cY)
                    a1 = cX-x
                    b1 = cY-y
                    pixel_centroid_distance = np.sqrt((a1*a1)+(b1*b1))
                    #a2 = np.absolute(x-(width/2))
                    #b2 = np.absolute(y-(height/2))
                    #If distance to another centroid is smaller than to the center, assign pixel to that centroid
                    if pixel_centroid_distance < smallest_distance:
                        smallest_distance = pixel_centroid_distance
                        X = cX
                        Y = cY
                image[x,y] = image[X,Y]
    '''
    for coords in centers:
        cX = coords[1]
        cY = coords[0]
        image[cX,cY] = 8 
    '''
    return image

def teeth_watershed_2(visualize = False):
    for idx, image in enumerate(watershedImgs):
        img = cv2.imread(data_dir+image)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #Original
        #ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) #otsu thresholding finds optimal thresholding value w.r.t. gray image histogram

        #New for teeth
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
        
        #Fill in-tooth holes
        im_floodfill = thresh.copy()
        h, w = thresh.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(im_floodfill, mask, (0,0), 255)
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        new_thresh = thresh | im_floodfill_inv
        if visualize:
            show_image(thresh, new_thresh)

        #Find out what parts are certainly teeth
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(new_thresh, cv2.MORPH_OPEN, kernel, iterations = 3)
        #Make sure that pixels far from the center of the tooth are background
        sure_bg = cv2.dilate(opening, kernel, iterations = 3)
        #Make sure that pixels openings near the center of the tooth are foreground

        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.2*dist_transform.max(), 255, 0)

        if visualize:
            show_image(sure_bg, sure_fg)
        #Compute again to remove narrow bridges between sure_fg
        sure_fg = np.uint8(sure_fg)
        dist_transform = cv2.distanceTransform(sure_fg, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.4*dist_transform.max(), 255, 0)
        #show_image(sure_bg, sure_fg)

        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        if visualize:
            show_image(sure_bg, sure_fg)

        #Mark background with 0, 
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers+1
        markers[unknown==255] = 0

        if visualize:
            plt.imshow(markers, cmap='jet')
            plt.show()

        print('datatype before: ', markers.dtype)
        with open('before.txt', 'w') as f:
            print(markers, file=f)
        markers = find_centroid(markers, idx)
        markers = np.int32(markers)
        with open('after.txt', 'w') as f:
            print(markers, file=f)
        print('datatype after: ', markers.dtype)

        if visualize:
            plt.imshow(markers, cmap='jet')
            plt.show()

        np.uint8(markers)
        markers = cv2.watershed(img, markers)
        img[markers==-1] = [255, 0, 0]

        height, width = np.shape(markers)
        label = findLabel(markers, width, height)
        labelList.append(label)
        watershedImageList.append(markers)

        #print('maxim', np.max(markerTemp))
        #find_centroid(markerTemp)

        if visualize:
            plt.imshow(markers, cmap='jet')
            plt.show()

        dist = markers.copy()
        dist = distance_check(dist, idx)
        distanceCheckedList.append(dist)

        if visualize:
            plt.imshow(dist, cmap='jet')
            plt.show()

def binaryImage(image, label):
    '''
    width, height = np.shape(image)
    for x in range(0, width):
        for y in range(0, height):
            if image[x,y] == label:
                image[x,y] = 1
            else:
                image[x,y] = 0
    return image
    '''
    maxim = np.max(image)
    minim = np.min(image)
    print(minim, maxim)
    image = np.uint8(image)
    ret1, thresh1 = cv2.threshold(image, label-1, maxim, cv2.THRESH_BINARY)
    ret2, thresh2 = cv2.threshold(image, label, maxim, cv2.THRESH_BINARY)
    final = (thresh1 != thresh2) #Take difference between the 2 thresholded images to be left with segmented tooth
    #plt.imshow(final, cmap='jet')
    #plt.show()
    return final

def processImage(visualize = False):
    for idx in range(0, len(watershedImageList)-1):
        image1 = watershedImageList[idx]
        image2 = distanceCheckedList[idx]

        label1 = labelList[idx]
        label2 = labelList[idx]

        bin_image1 = binaryImage(image1, label1)
        bin_image2 = binaryImage(image2, label2)

        '''
        plt.imshow(image1, cmap='jet')
        plt.show()
        plt.imshow(image2, cmap='jet')
        plt.show()
        '''

        final_image = (bin_image1 == bin_image2)
        if visualize:
            print('no overlap binary images')
            plt.imshow(final_image, cmap='jet')
            plt.show()
            print('initial image watershed')
            plt.imshow(image1, cmap='jet')
            plt.show()
            print('initial image distance')
            plt.imshow(image2, cmap='jet')
            plt.show()

        with open('watershedImg.txt', 'w') as f:
            print(image1, file=f)

        temp_image = final_image.copy()
        height, width = np.shape(temp_image)
        for x in range(height):
            for y in range(width):
                if (image1[x,y] != label1):
                    temp_image[x,y] = False

        final_image = (temp_image == final_image)
        final_image = final_image.astype(int)
        temp_image = temp_image.astype(int)
        #print('final image')
        #plt.imshow(temp_image, cmap='jet')
        #plt.show()
        binary_slices_for_model.append(temp_image)
        np.save('binary_tooth_slices.npy', binary_slices_for_model)


teeth_watershed_2(visualize = True)
#adjust_masks()
processImage(visualize = True)
#construct3DModel()
#find_centroid()
#Find Contours











def teeth_watershed():
    for image in watershedImgs:
        img = cv2.imread(data_dir+image)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #Original
        #ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) #otsu thresholding finds optimal thresholding value w.r.t. gray image histogram

        #New for teeth
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
        #show_image(ret, thresh)

        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)
        #Make sure that pixels far from the center of the tooth are background
        sure_bg = cv2.dilate(opening, kernel, iterations = 3)
        #Make sure that pixels openings near the center of the tooth are foreground
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.3*dist_transform.max(), 255, 0)

        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        show_image(sure_bg, sure_fg)

        #Cut narrow sections in sure_fg


        #Mark background with 0, 
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers+1
        markers[unknown==255] = 0

        plt.imshow(markers, cmap='jet')
        plt.show()

        markers = cv2.watershed(img, markers)
        img[markers==-1] = [255, 0, 0]

        plt.imshow(markers, cmap='jet')
        plt.show()


def test_coins():
    img = cv2.imread('cropped/coins/coins.jpg')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #Original
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) #otsu thresholding finds optimal thresholding value w.r.t. gray image histogram

    #New for teeth
    #ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)

    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)
    #Make sure that pixels far from the center of the tooth are background
    sure_bg = cv2.dilate(opening, kernel, iterations = 3)
    #Make sure that pixels openings near the center of the tooth are foreground
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    show_image(sure_bg, sure_fg)

    #Mark background with 0, 
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers+1
    markers[unknown==255] = 0

    plt.imshow(markers, cmap='jet')
    plt.show()

    markers = cv2.watershed(img, markers)
    img[markers==-1] = [255, 0, 0]

    plt.imshow(markers, cmap='jet')
    plt.show()