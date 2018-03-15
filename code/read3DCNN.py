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

#test = axial_df['patient'][1]


#print(labels_df[1].tolist())

# for patient in patients[:15]: #only one patient
#     label = labels_df.at[patient, 1]
#     print('label is:', label)
#     path = data_dir + patient #path to the patient
#     slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)] # Get all dicom files
#     slices.sort(key = lambda x: int(x.ImagePositionPatient[2])) # X refers to dicom file
#     print(len(slices), label)
#     print('The dimensions are (l*b*h): {} * {} * {}'.format(slices[0].Rows, slices[0].Columns, len(slices)))
#     print('\n')


######RESIZING IMAGE#####
from matplotlib import pyplot as plt
import cv2
import numpy as np
import math

IMG_RESIZE = 50 #Pixel dimensions of axial slices after resize
NR_SLICES = 20 #NR of axial slices used for one volume
KERNEL_SIZE = 50

store_path = '/media/ruben/Seagate Expansion Drive/bachelorProject/data/rewritten/' #'/media/ruben/Seagate Expansion Drive/bachelorProject/data/rewritten/'
patCount = 0
numPatients = 15

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n] #returns a generator, you can only loop once over generators

def mean(l):
    return sum(l)/len(l)

#for i, patient in enumerate(patients[:numPatients]):
def process_data(patient, labels_df, IMG_RESIZE=50, NR_SLICES=20, visualize=False):
    label = int(labels_df[1][patient])
    print(patient)
    folder = patient
    cur_dir = data_dir+folder

    slices = [dicom.read_file(cur_dir+'/'+s) for s in os.listdir(cur_dir)] #Get all dicom files for patient
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2])) #X refers to dicom file

    new_slices = []
    slices = [cv2.resize(np.array(each_slice.pixel_array), (IMG_RESIZE,IMG_RESIZE)) for each_slice in slices]
    chunk_sizes = math.ceil(len(slices)/NR_SLICES) #Calculate how large the chunks will be
    for slice_chunk in chunks(slices, chunk_sizes):
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        new_slices.append(slice_chunk)

    #Make sure that the length of new slices = NR_SLICES
    if len(new_slices) == NR_SLICES-1:
        new_slices.append(new_slices[-1])
    if len(new_slices) == NR_SLICES-2:
        new_slices.append(new_slices[-1])
        new_slices.append(new_slices[-1])
    if len(new_slices) == NR_SLICES+2:
        new_val = list(map(mean, zip(*[new_slices[NR_SLICES-1], new_slices[NR_SLICES]])))
        del new_slices[NR_SLICES]
        new_slices[NR_SLICES-1] = new_val
    if len(new_slices) == NR_SLICES+1:
        new_val = list(map(mean, zip(*[new_slices[NR_SLICES-1], new_slices[NR_SLICES]])))
        del new_slices[NR_SLICES]
        new_slices[NR_SLICES-1] = new_val

    print(len(slices), len(new_slices))

    if visualize:
        fig = plt.figure()
        for num, each_slice in enumerate(new_slices):
            y = fig.add_subplot(4,5,num+1)
            #new_image = cv2.resize(np.array(each_slice.pixel_array), (IMG_RESIZE,IMG_RESIZE))
            plt.imshow(each_slice)
        plt.show()

    #Make cleaner code in future
    print(label)
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

    return np.array(new_slices), label

much_data = []
for num, patient in enumerate(patients):
    # print('pat:', patient)
    # if num%100==0:
    #     print('num:', num)

    try:
        img_data,label = process_data(patient,labels_df,IMG_RESIZE=IMG_RESIZE, NR_SLICES=NR_SLICES, visualize=True)
        much_data.append([img_data,label])
    except KeyError as e:
        print('Data has no label')

    np.save('muchdata-{}-{}-{}.npy'.format(IMG_RESIZE,IMG_RESIZE,NR_SLICES), much_data)

################TRASH###################
    # patCount += 1
    # teethSliceLow = int(axial_df['lowerThresh'][i+1])
    # teethSliceHigh = int(axial_df['upperThresh'][i+1])
    # for slice in range(0,len(slices)):
    #     image = np.array(slices[slice].pixel_array)
    #     width, height = np.shape(image)
    #     #image = cv2.resize(np.array(slices[50].pixel_array), (IMG_RESIZE,IMG_RESIZE))
    #     # plt.imshow(image, cmap='gray')
    #     # plt.show()
    #     dims = int(width/KERNEL_SIZE)
    #     nrSteps = dims * dims
    #     for idx in range(0,nrSteps):
    #         rowIDX = int(idx/int(width/KERNEL_SIZE))
    #         colIDX = idx%dims
    #         newimg = image[(rowIDX*KERNEL_SIZE):(rowIDX*KERNEL_SIZE)+KERNEL_SIZE, (colIDX*KERNEL_SIZE):(colIDX*KERNEL_SIZE)+KERNEL_SIZE]
    #         total = (newimg.sum(-1)).sum(-1)
    #         if(total != 0):
    #             if(teethSliceLow <= slice <= teethSliceHigh):
    #                 fileName = 'pat1/teeth/img'+str(slice)+'_'+str(idx)+'.png'
    #             else:
    #                 fileName = 'pat1/noTeeth/img'+str(slice)+'_'+str(idx)+'.png'
    #             directory = os.path.join(store_path,fileName)
    #             cv2.imwrite(directory, newimg) #Save new image
            #fig.add_subplot(dims,dims,idx+1)
            #plt.imshow(newimg)
        #plt.show()
    # fig = plt.figure()
    # for num, each_slice in enumerate(slices[77:97]):
    #     y = fig.add_subplot(4,5,num+1)
    #     new_image = cv2.resize(np.array(each_slice.pixel_array),(IMG_RESIZE,IMG_RESIZE)) #Downsize original slice
    #     plt.imshow(new_image, cmap='gray')
    # plt.show()

