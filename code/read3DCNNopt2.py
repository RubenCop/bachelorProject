import dicom
import os
import re
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

IMG_RESIZE = 50             #Pixel dimensions of axial slices after resize, NOT USED AT THE MOMENT
NR_SLICES = 10              #NR of axial slices used for one volume
SUB_IMG_SIZE = 50           #So the final volume of one example becomes SUB_IMG_SIZE*SUB_IMG_SIZE*NR_SLICES
REDUNDANCY_THRESH = 1000 #Adjust if volume of examples is altered

#store_path = '/media/ruben/Seagate Expansion Drive/bachelorProject/data/rewritten/' #'/media/ruben/Seagate Expansion Drive/bachelorProject/data/rewritten/'
patCount = 0
numPatients = 15

totalList = []

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n] #returns a generator, you can only loop once over generators

def mean(l):
    return sum(l)/len(l)

def redundancy(list):
    for idx in range(0, len(list)):
        total = (list[idx].sum(-1)).sum(-1)
        if total > REDUNDANCY_THRESH:
            totalList.append(total)
            return False
        else:
            return True


#for i, patient in enumerate(patients[:numPatients]):
def process_data(num, patient, labels_df, IMG_RESIZE=50, NR_SLICES=20, include_all_slices = True, visualize=False):
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

    lowerTeethSlice = int(axial_df['lowerThresh'][int(re.search(r'\d+', str(patient)).group())])
    upperTeethSlice = int(axial_df['upperThresh'][int(re.search(r'\d+', str(patient)).group())])
    #print("upper boundary: ", upperTeethSlice, "lower boundary: ", lowerTeethSlice)

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
        #print('new')
        #for i in range(0, int(width/SUB_IMG_SIZE)):
        for i in range(0, NR_SLICES):
            #print(len(slices), count+i, width/SUB_IMG_SIZE)

            image = np.array(slices[count+i].pixel_array) #get every NR_SLICES images
            width, height = np.shape(image)
            dims = int(width/SUB_IMG_SIZE)
            nrSteps = dims*dims
            #Cut single slice into squares
            for idx in range(0, nrSteps):
                rowIDX = int(idx/dims)
                colIDX = idx%dims
                newimg = image[(rowIDX*SUB_IMG_SIZE) : (rowIDX*SUB_IMG_SIZE)+SUB_IMG_SIZE, (colIDX*SUB_IMG_SIZE) : (colIDX*SUB_IMG_SIZE)+SUB_IMG_SIZE]
                temp.append(newimg) #Temp contains squares of one slice
            imgList.append(temp) #imgList contains squares of 10 slices in the end
            temp = []
        #print(np.shape(imgList)) #NR slices, NR images, width image, height image
        #print('dims: ', dims)
        #count = 0
        for idx in range(0, dims*dims):
            for imgListidx in range(0, NR_SLICES):
                #count+=1
                newList.append(imgList[imgListidx][idx]) #append NR_SLICES images to newList
            new_slices.append(newList) #shape 1, 10, 50, 50
            newList = []
            #print(np.shape(new_slices))
            #for slice_chunk in chunks(new_slices, NR_SLICES):
            #slice_chunk = list(map(mean, zip(*new_slices[0]))) # new_slices[0] has shape 10, 50, 50, so 10 slices of sub block
            #print('testshape', np.shape(new_slices[0]))
            #print('slice chunk:', np.shape(slice_chunk))
            #print(np.shape(new_slices))
            #final_slices.append(slice_chunk)
            if (not redundancy(new_slices[0])):
                #print('added chunk!')
                if(lowerTeethSlice <= count <= upperTeethSlice) and not include_all_slices:
                    final_slices.append(new_slices[0])
                elif(include_all_slices):
                    final_slices.append(new_slices[0])
                if visualize and (lowerTeethSlice <= count <= upperTeethSlice):
                    fig = plt.figure()
                    for num, each_slice in enumerate(new_slices[0]):
                        y = fig.add_subplot(2, 5, num+1)
                        plt.imshow(each_slice)
                    plt.show()
            else:
                #print('skipped chunk!')
                pass
            #if len(temp_chunk) > 0:
                #print('len temp chunk:', len(temp_chunk))
            new_slices = []
        #print('count', count)
        imgList = []

    #print(len(slices), len(final_slices))

    #Make cleaner code in future
    #print('label: ', label)
    # if 0<=label<1100: 
    #     label = np.array([0,1])
    #     print('yes')
    # elif 1100<=label<1800: 
    #     label = np.array([1,0])
    #     print('no')

    #len labels = 14
    if 0<=label<600: label = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1])
    elif 600<=label<700: label = np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0])
    elif 700<=label<800: label = np.array([0,0,0,0,0,0,0,0,0,0,0,1,0,0])
    elif 800<=label<900: label = np.array([0,0,0,0,0,0,0,0,0,0,1,0,0,0])
    elif 900<=label<1000: label = np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0])
    elif 1000<=label<1100: label = np.array([0,0,0,0,0,0,0,0,1,0,0,0,0,0])
    elif 1100<=label<1200: label = np.array([0,0,0,0,0,0,0,1,0,0,0,0,0,0])
    elif 1200<=label<1300: label = np.array([0,0,0,0,0,0,1,0,0,0,0,0,0,0])
    elif 1300<=label<1400: label = np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0])
    elif 1400<=label<1500: label = np.array([0,0,0,0,1,0,0,0,0,0,0,0,0,0])
    elif 1500<=label<1600: label = np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0])
    elif 1600<=label<1700: label = np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0])
    elif 1700<=label<1800: label = np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0])
    else: label = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0])

    fig = plt.figure()

    print('final slices, numpy shape:', np.shape(final_slices))
    #print('MEAN: ', math.ceil(np.mean(totalList)))

    return np.array(final_slices), label


for num, patient in enumerate(patients):
    data_3d = []
    print('\n')
    # print('pat:', patient)
    # if num%100==0:
    #     print('num:', num)
    # if num > 0: #Get data of first x patients
    #     break
    try:
        img_data,label = process_data(num, patient,labels_df,IMG_RESIZE=IMG_RESIZE, NR_SLICES=NR_SLICES, include_all_slices = False, visualize=False)
        [data_3d.append([item,label]) for item in img_data] #original: data_3d.append([img_data,label])
        #This appends every sub image as a separate data point (so multiple datapoints from 1 patient)
        print(np.shape(data_3d))
    except KeyError as e:
        print('Data has no label')
    np.save(os.path.join('/media/ruben/Seagate Expansion Drive/bachelorProject/code/data/dentalArea','3dData-PAT'+str(num)+'-{}-{}-{}.npy'.format(SUB_IMG_SIZE,SUB_IMG_SIZE,NR_SLICES)), data_3d)
    print('Data saved in: 3dData-PAT'+str(num)+'-{}-{}-{}.npy'.format(SUB_IMG_SIZE, SUB_IMG_SIZE, NR_SLICES))
