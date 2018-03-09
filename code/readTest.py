import dicom
import os
import numpy
from matplotlib import pyplot, cm

FILESTR= "pat"
PATH = "/media/ruben/Seagate Expansion Drive/bachelorProject/data/final/"
filesDict = {}

def read_patient_data(folder):
    filesList = []
    for dirName, subdirList, fileList in os.walk(PATH+folder):
        print('Directory location:\n', dirName)
        print('Amount of slices in directory: ', len(fileList), '\n')
        for filename in fileList:
            if '.dcm' in filename.lower():
                filesList.append(os.path.join(dirName,filename))
    filesDict.update({folder:filesList})


numPatients = 0
for dirnames in os.walk(PATH):
    numPatients += 1
print("number of patients: ", numPatients-1)

for i in range(1, 5):
    folder = FILESTR+str(i)
    read_patient_data(folder)

for key, values in filesDict.items():
    #print('key:', key, '\n, values', values)
    pass

RefDs = dicom.read_file(filesDict['pat1'][0])
#Print dicom file specific data
#print(RefDs)
#RefDs = dicom.read_file("/media/ruben/Seagate Expansion Drive/bachelorProject/data/final/pat1/pat1 (300).dcm")
ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(filesDict['pat1']))
ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), 
    float(RefDs.PixelSpacing[1]), 
    float(RefDs.SliceThickness))

x = numpy.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
y = numpy.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
z = numpy.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])

# The array is sized based on 'ConstPixelDims'
ArrayDicom = numpy.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

# loop through all the DICOM files
for filenameDCM in filesDict['pat1']:
    # read the file
    ds = dicom.read_file(filenameDCM)
    # store the raw image data
    ArrayDicom[:, :, filesDict['pat1'].index(filenameDCM)] = ds.pixel_array  


#plot the dicom slice
pyplot.figure(dpi=300)
pyplot.axes().set_aspect('equal', 'datalim')
pyplot.set_cmap(pyplot.gray())
for i in range(200, 210): #show first 10 slices
    pyplot.pcolormesh(x, y, numpy.flipud(ArrayDicom[:, :, i]))
    pyplot.show()

