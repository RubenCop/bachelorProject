import dicom
import os
import pandas as pd

data_dir = "/media/ruben/Seagate Expansion Drive/bachelorProject/data/final/" #"/media/ruben/Seagate Expansion Drive/bachelorProject/data/dentsplySirona/testData/"  #"/media/ruben/Seagate Expansion Drive/bachelorProject/data/final/"
patients = os.listdir(data_dir)
#Read labels from .txt file
labels_df = pd.read_csv("labels.txt", sep=' ', header=None)
labels_df = labels_df.set_index([0])
#print(labels_df.head())
#print('\nNumber of patients: ', len(patients))
#print(labels_df[1]['pat1'])

axial_df = pd.read_csv("axialteeth.txt", sep=' ', header=None)
#axial_df = axial_df.set_index([0]) #Set first column as row names

header = axial_df.iloc[0]
axial_df = axial_df[1:] #remove first row
axial_df.columns = header


######RESIZING IMAGE#####
from matplotlib import pyplot as plt
import cv2
import numpy as np
import math

IMG_RESIZE = 500 #Pixel dimensions of axial slices after resize
NR_SLICES = 10 #NR of axial slices used for one volume
SUB_IMG_SIZE = 50
REDUNDANCY_THRESH = 0

#store_path = '/media/ruben/Seagate Expansion Drive/bachelorProject/data/rewritten/' #'/media/ruben/Seagate Expansion Drive/bachelorProject/data/rewritten/'
patCount = 0
numPatients = 15

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n] #returns a generator, you can only loop once over generators

def mean(l):
    return sum(l)/len(l)

def redundancy(list):
    for idx in range(0, len(list)):
        total = (list[idx].sum(-1)).sum(-1)
        if total > REDUNDANCY_THRESH:
            return True
        else:
            return False


#for i, patient in enumerate(patients[:numPatients]):
def process_data(num, patient, labels_df, IMG_RESIZE=50, NR_SLICES=20, visualize=False):
    label = int(labels_df[1][patient])
    print(patient)
    folder = patient
    cur_dir = data_dir+folder

    slices = [dicom.read_file(cur_dir+'/'+s) for s in os.listdir(cur_dir)] #Get all dicom files for patient
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2])) #X refers to dicom file

    #Old code
    # new_slices = []
    # slices = [cv2.resize(np.array(each_slice.pixel_array), (IMG_RESIZE,IMG_RESIZE)) for each_slice in slices]
    # chunk_sizes = math.ceil(len(slices)/NR_SLICES) #Calculate how large the chunks will be
    # for slice_chunk in chunks(slices, chunk_sizes):
    #     slice_chunk = list(map(mean, zip(*slice_chunk)))
    #     new_slices.append(slice_chunk)
    

    ##############################

    lowerTeethSlice = int(axial_df['lowerThresh'][num+1])
    upperTeethSlice = int(axial_df['upperThresh'][num+1])
    print("upper boundary: ", upperTeethSlice, "lower boundary: ", lowerTeethSlice)

    new_slices = []
    temp = []
    imgList = []
    newList = []
    final_slices = []
    #slices = [cv2.resize(np.array(each_slice.pixel_array), (IMG_RESIZE,IMG_RESIZE)) for each_slice in slices]
    
    #To get dimensions of single image
    image = np.array(slices[0].pixel_array)
    width, height = np.shape(image)
    print('width:', width, 'height: ', height)
    print(len(slices))
    print('nr steps: ', int(len(slices)/NR_SLICES))
    for count in range(0, int(len(slices)/NR_SLICES)*NR_SLICES, NR_SLICES): #stepsize = NR_SLICES, last slices are thrown away
        for i in range(0, int(width/SUB_IMG_SIZE)):
            image = np.array(slices[count+i].pixel_array)
            width, height = np.shape(image)
            dims = int(width/SUB_IMG_SIZE)
            nrSteps = dims*dims
            #Cut single slice into squares
            for idx in range(0, nrSteps):
                rowIDX = int(idx/dims)
                colIDX = idx%dims
                newimg = image[(rowIDX*SUB_IMG_SIZE) : (rowIDX*SUB_IMG_SIZE)+SUB_IMG_SIZE, (colIDX*SUB_IMG_SIZE) : (colIDX*SUB_IMG_SIZE)+SUB_IMG_SIZE]
                temp.append(newimg) #Temp contains squares of one slice
            imgList.append(temp) #imgList contains squares of 10 slices
            temp = []
        #print(np.shape(imgList)) #NR slices, NR images, width image, height image
        #print('dims: ', dims)
        count = 0
        for idx in range(0, dims*dims):
            for imgListidx in range(0, NR_SLICES):
                count+=1
                newList.append(imgList[imgListidx][idx]) #append NR_SLICES images to newList
            new_slices.append(newList)
            newList = []
            #print(np.shape(new_slices))
            #for slice_chunk in chunks(new_slices, NR_SLICES):
            slice_chunk = list(map(mean, zip(*new_slices[0])))
            #print('slice chunk:', np.shape(slice_chunk))
            #print(np.shape(new_slices))
            #final_slices.append(slice_chunk)
            if not redundancy(new_slices[0]):
                print('added chunk!')
                final_slices.append(slice_chunk)
            else:
                print('skipped chunk!')
                #print('THROWN AWAY')
            new_slices = []
        print('count', count)
        imgList = []

    print(len(slices), len(final_slices))

    if visualize:
        fig = plt.figure()
        for num, each_slice in enumerate(new_slices):
            y = fig.add_subplot(4,5,num+1)
            #new_image = cv2.resize(np.array(each_slice.pixel_array), (IMG_RESIZE,IMG_RESIZE))
            plt.imshow(each_slice)
        plt.show()

    #Make cleaner code in future
    print('label: ', label)
    # if 0<=label<1100: 
    #     label = np.array([0,1])
    #     print('yes')
    # elif 1100<=label<1800: 
    #     label = np.array([1,0])
    #     print('no')

    if 0<=label<600: label = np.array([0,0,0,0,0,0,1])
    elif 600<=label<800: label = np.array([0,0,0,0,0,1,0])
    elif 800<=label<1000: label = np.array([0,0,0,0,1,0,0])
    elif 1000<=label<1200: label = np.array([0,0,0,1,0,0,0])
    elif 1200<=label<1400: label = np.array([0,0,1,0,0,0,0])
    elif 1400<=label<1600: label = np.array([0,1,0,0,0,0,0])
    elif 1600<=label<1800: label = np.array([1,0,0,0,0,0,0])

    fig = plt.figure()

    print(np.shape(final_slices))

    return np.array(final_slices), label

much_data = []
for num, patient in enumerate(patients):
    # print('pat:', patient)
    # if num%100==0:
    #     print('num:', num)
    if num > 0: #Get data of first x patients
        break
    try:
        img_data,label = process_data(num, patient,labels_df,IMG_RESIZE=IMG_RESIZE, NR_SLICES=NR_SLICES, visualize=False)
        much_data.append([img_data,label])
    except KeyError as e:
        print('Data has no label')

    np.save('muchdata-{}-{}-{}.npy'.format(SUB_IMG_SIZE,SUB_IMG_SIZE,NR_SLICES), much_data)
    print('Data saved in: muchdata-{}-{}-{}.npy'.format(SUB_IMG_SIZE, SUB_IMG_SIZE, NR_SLICES))
