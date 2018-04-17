import dicom
import os
import pandas as pd
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt


#READ LABEL FILE
data_dir = "/media/ruben/Seagate Expansion Drive/bachelorProject/data/final/"
patients = os.listdir(data_dir)
labels_df = pd.read_csv("labels.txt", sep=' ', header=None)
labels_df = labels_df.set_index([0])
axial_df = pd.read_csv("axialteeth.txt", sep=' ', header=None)
header = axial_df.iloc[0]
axial_df = axial_df[1:] #First row is removed
axial_df.columns = header

##################################################
IMG_RESIZE = 50
SUB_IMG_SIZE = 50

patCount = 0
numPatients = 15

def process_data(num, patient, labels_df, IMG_RESIZE=50, include_all_slices = True, visualize=False):
    label = int(labels_df[1][patient])
    print(patient)
    folder = patient
    cur_dir = data_dir+folder

    slices = [dicom.read_file(cur_dir+'/'+s) for s in os.listdir(cur_dir)] #Get all dicom files for patient
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2])) #X refers to dicom file
    print(len(slices))
    new_slices = []

    lowerTeethSlice = int(axial_df['lowerThresh'][num+1])
    upperTeethSlice = int(axial_df['upperThresh'][num+1])
    print("upper boundary: ", upperTeethSlice, "lower boundary: ", lowerTeethSlice)

    for slice in range(0, len(slices)):
        image = np.array(slices[slice].pixel_array)
        width, height = np.shape(image)
        dims = int(width/SUB_IMG_SIZE)
        nrSteps = dims * dims #NR of steps needed for each slice
        for idx in range(0, nrSteps):
            rowIDX = int(idx/int(width/SUB_IMG_SIZE))
            colIDX = idx%dims
            newimg = image[(rowIDX*SUB_IMG_SIZE) : (rowIDX*SUB_IMG_SIZE)+SUB_IMG_SIZE, (colIDX*SUB_IMG_SIZE) : (colIDX*SUB_IMG_SIZE)+SUB_IMG_SIZE]
            print('shape', np.shape(newimg))
            total = (newimg.sum(-1)).sum(-1) #Get total pixel value
            if(total != 0):
                #ONLY use data in the dental region. Including other data leads to memory error, too much memory needed
                if(lowerTeethSlice <= slice <= upperTeethSlice) and not include_all_slices:
                    new_slices.append(newimg)
                elif(include_all_slices):
                    new_slices.append(newimg)

    if visualize:
        fig = plt.figure()
        for num, each_slice in enumerate(new_slices):
            y = fig.add_subplot(4,5,num+1)
            plt.imshow(each_slice)
        plt.show()

    print(label)

    # for slice in range(0,numPatient+1):
    #     image = np.array(slices[slice].pixel_array)
    #     width, height = np.shape(image)
    #     dims = int(width/KERNEL_SIZE)
    #     nrSteps = dims*dims
    #     for idx in range(0, nrSteps):
    #         rowIDX = int(idx/int(widht/KERNEL_SIZE))
    #         colIDX = idx%dims
    #         newimg = image[(rowIDX*KERNEL_SIZE):(rowIDX*KERNEL_SIZE)+KERNEL_SIZE, (colIDX*KERNEL_SIZE):(colIDX*KERNEL_SIZE)+KERNEL_SIZE]
    #         total = 

    if 0<=label<600: label = np.array([0,0,0,0,0,0,1])
    elif 600<=label<800: label = np.array([0,0,0,0,0,1,0])
    elif 800<=label<1000: label = np.array([0,0,0,0,1,0,0])
    elif 1000<=label<1200: label = np.array([0,0,0,1,0,0,0])
    elif 1200<=label<1400: label = np.array([0,0,1,0,0,0,0])
    elif 1400<=label<1600: label = np.array([0,1,0,0,0,0,0])
    elif 1600<=label<1800: label = np.array([1,0,0,0,0,0,0])

    return np.array(new_slices), label

data_2d = []
for num, patient in enumerate(patients):
    try:
        img_data, label = process_data(num, patient, labels_df, IMG_RESIZE=IMG_RESIZE, include_all_slices = False, visualize=False)
        data_2d.append([img_data, label])
    except KeyError as e:
        print('Data has no label')
    np.save('2dData-{}-{}.npy'.format(IMG_RESIZE, IMG_RESIZE), data_2d)