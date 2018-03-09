import dicom
import os
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

IMG_RESIZE = 50 #Pixels after resize
KERNEL_SIZE = 50
store_path = '/media/ruben/Seagate Expansion Drive/bachelorProject/data/test/' #'/media/ruben/Seagate Expansion Drive/bachelorProject/data/rewritten/'
patCount = 0
numPatients = 1

def readAndChop():
    data_dir = "/media/ruben/Seagate Expansion Drive/bachelorProject/data/dentsplySirona/testData/"  #"/media/ruben/Seagate Expansion Drive/bachelorProject/data/final/"
    patients = os.listdir(data_dir)
    labels_df = pd.read_csv("labels.txt", sep=' ', header=None)
    labels_df = labels_df.set_index([0])
    axial_df = pd.read_csv("axialteeth.txt", sep=' ', header=None)
    header = axial_df.iloc[0]
    axial_df = axial_df[1:] #remove first row
    axial_df.columns = header

    for i, patient in enumerate(patients[:numPatients]):
        teethSliceLow = int(axial_df['lowerThresh'][i+1])
        teethSliceHigh = int(axial_df['upperThresh'][i+1])
        patCount += 1
        folder = 'pat'+str(patCount)
        cur_dir = data_dir+folder

        slices = [dicom.read_file(cur_dir+'/'+s) for s in os.listdir(cur_dir)] #Get all dicom files for patient
        slices.sort(key = lambda x: int(x.ImagePositionPatient[2])) #X refers to dicom file

        #fig = plt.figure()
        for slice in range(0,len(slices)):
            image = np.array(slices[slice].pixel_array)
            width, height = np.shape(image)
            #image = cv2.resize(np.array(slices[50].pixel_array), (IMG_RESIZE,IMG_RESIZE))
            # plt.imshow(image, cmap='gray')
            # plt.show()
            dims = int(width/KERNEL_SIZE)
            nrSteps = dims * dims
            for idx in range(0,nrSteps):
                rowIDX = int(idx/int(width/KERNEL_SIZE))
                colIDX = idx%dims
                newimg = image[(rowIDX*KERNEL_SIZE):(rowIDX*KERNEL_SIZE)+KERNEL_SIZE, (colIDX*KERNEL_SIZE):(colIDX*KERNEL_SIZE)+KERNEL_SIZE]
                total = (newimg.sum(-1)).sum(-1)
                if(total != 0):
                    if(teethSliceLow <= slice <= teethSliceHigh):
                        fileName = 'pat1/teeth/img'+str(slice)+'_'+str(idx)+'.png'
                    else:
                        fileName = 'pat1/noTeeth/img'+str(slice)+'_'+str(idx)+'.png'
                    directory = os.path.join(store_path,fileName)
                    cv2.imwrite(directory, newimg) #Save new image