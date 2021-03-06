import dicom
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import pygame, sys
import scipy.misc
pygame.init()
np.set_printoptions(threshold=np.nan)

data_dir = "/media/ruben/Seagate Expansion Drive/bachelorProject/data/final/"
patients = os.listdir(data_dir)
whole_img_save_dir = 'tempData/'
cropped_img_save_dir = 'cropped/teeth/'
volume_img_save_dir = 'cropped/teeth/volume/'

START_SLICE = 125
NUMBER_OF_SLICES = 20
RESIZE_FACTOR = 2
THRESHOLD = 2000

def displayImage(screen, px, topleft, prior):
    x, y = topleft
    width = pygame.mouse.get_pos()[0] - topleft[0]
    height = pygame.mouse.get_pos()[1] - topleft[1]
    if width < 0:
        x += width
        width = abs(width)
    if height < 0:
        y += height
        height = abs(height)
    circleX = int((width)/2)
    circleY = int((height)/2)
    #print(x, y, width, height, circleX, circleY)
    current = x, y, width, height
    if not (width and height):
        return current
    if current == prior:
        return current

    #Draw transparent box over to-be-cropped part of the image
    screen.blit(px, px.get_rect())
    im = pygame.Surface((width, height))
    im.fill((128, 128, 128)) #grey color over box
    pygame.draw.rect(im, (32, 32, 32), im.get_rect(), 1)
    pygame.draw.circle(im, (255, 0, 0), (circleX, circleY), 5, 0)
    im.set_alpha(128)
    screen.blit(im, (x,y))
    pygame.display.flip()

    return(x, y, width, height)

def setup(image):
    #px = pygame.image.load('tempData/outfile.jpg')
    img_name = os.listdir(whole_img_save_dir)[0]
    px = pygame.image.load('tempData/'+str(img_name))
    screen = (pygame.display.set_mode((px.get_rect()[2], px.get_rect()[3])))
    screen.blit(px, px.get_rect())
    pygame.display.flip()
    return screen, px

def mainLoop(screen, px):
    topleft = bottomright = prior = None
    n=0
    while n!=1:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONUP:
                if not topleft:
                    topleft = event.pos
                else:
                    bottomright = event.pos
                    n=1
        if topleft:
            prior = displayImage(screen, px, topleft, prior)
    return(topleft+bottomright)

def cropImg(image, image_path, first_crop = False, left=0, upper=0, right=0, lower=0):
    #print(left, upper, right, lower)
    if(first_crop):
        screen, px = setup(image)
        left, upper, right, lower = mainLoop(screen, px)

        if right<left:
            left, right = right, left
        if lower < upper:
            lower, upper = upper, lower
        im = Image.open(image_path)
        im = im.crop((left, upper, right, lower))
        pygame.display.quit()
        return im, left, upper, right, lower
    else:
        im = Image.open(image_path)
        im = im.crop((left, upper, right, lower))
        return im

def saveMultipleSlices(slice_list):
    for num, image in enumerate(slice_list):
        scipy.misc.imsave(whole_img_save_dir+'img'+str(num)+'.jpg', image)

def threshold_func(image, threshold):
    width, height = np.shape(image)
    for x in range(0, width):
        for y in range(0, height): 
            if image[x,y] < threshold:
                image[x,y] = 0
    return image

def getMultipleSlices(patient, start_slice = 0, number_of_slices = 10):
    output_list = []

    current_directory = data_dir+patient
    slices = [dicom.read_file(current_directory+'/'+s) for s in os.listdir(current_directory)]  #Get all dicom images
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))  #X refers to the dicom file

    '''
    for num, slice in enumerate(slices):
        slice = slice.pixel_array
        if num == 130:
            print(slice)
            plt.imshow(slice, cmap='jet')
            plt.show()
    '''

    #Plot the slices
    fig, ax = plt.subplots()
    idx = 0
    for slice in range(1, len(slices)):
        if start_slice < slice < (start_slice+number_of_slices):
            idx += 1
            plt.subplot(4, 5, idx)
            plt.imshow(slices[slice].pixel_array)
    plt.show()

    if(start_slice+number_of_slices)>len(slices):
        print('Range of slices exceeds the number of slices that this scan contains')
        exit()

    for slice in range(0, len(slices)):
        image = np.array(slices[slice].pixel_array)
        #print(type(image))
        #ret, image = cv2.threshold(image, threshold, np.max(image), cv2.THRESH_TRUNC)
        #print(type(image))
        
        if(start_slice <= slice <= (start_slice+number_of_slices)):# and (slice < start_slice + number_of_slices):
            image = threshold_func(image, THRESHOLD)
            if(slice==start_slice):
                plt.imshow(image, cmap='jet')
                plt.show()
            width, height = np.shape(image)
            image = cv2.resize(image, (width*RESIZE_FACTOR, height*RESIZE_FACTOR))
            output_list.append(image)
    saveMultipleSlices(output_list)
    return output_list

'''
for patient in patients:
    image = getImage(patient)
    break

img = cropImg(image)
img.save(cropped_img_save_dir+'temp.png')
'''

for patient in patients:
    print(patient)
    image_list = getMultipleSlices(patient, START_SLICE, NUMBER_OF_SLICES)
    break


#Set flag, gets dimensions of first cropped image and than crops the following images to same size
first_crop = True
image_dirs = os.listdir(whole_img_save_dir)
print('Make sure the RED DOT is ON THE TOOTH that needs to be segmented')
for num, image_path in enumerate(image_dirs):
    if(first_crop):
        img, left, upper, right, lower = cropImg(image_list[0], 'tempData/'+image_path, first_crop)
    else:
        img = cropImg(image_list[0], 'tempData/'+image_path, first_crop, left, upper, right, lower)
    img.save(volume_img_save_dir+'img'+str(num)+'.png')
    first_crop = False







#################################################################
def getImage(patient):

    current_directory = data_dir+patient
    slices = [dicom.read_file(current_directory+'/'+s) for s in os.listdir(current_directory)]  #Get all dicom images
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))  #X refers to the dicom file

    for slice in range(0, len(slices)):
        image = np.array(slices[slice].pixel_array)

        if(slice % 10 == 0):
            pass
            #print('Slice number: ', slice)
            #fig = plt.figure()
            #plt.imshow(image)
            #plt.show()
    used_image = slices[130].pixel_array
    width, height = np.shape(used_image)
    used_image = cv2.resize(used_image, (width*2, height*2))
    scipy.misc.imsave(whole_img_save_dir+'outfile.jpg', used_image)
    #Take slice #130, nice teeth example
    return used_image