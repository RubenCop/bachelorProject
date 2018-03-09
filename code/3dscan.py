#To have a look at the 3D model; GOTO
#https://plot.ly/~RubenMACHL

import numpy as np
import dicom
import os
import matplotlib.pyplot as plt
from glob import glob
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage
from skimage import morphology 
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.graph_objs import *
import plotly.plotly as py
import plotly.figure_factory as FF
import plotly.tools
init_notebook_mode(connected=True) 

#Authenticate for 3D-model
plotly.tools.set_credentials_file(username='RubenMACHL', api_key='TJZwNrVdWuUk5YMuBcLR')

#Constants
PATH =  "/media/ruben/Seagate Expansion Drive/bachelorProject/data/dentsplySirona/testData/" #"/media/ruben/Seagate Expansion Drive/bachelorProject/data/final/"
O_PATH = working_path = "/media/ruben/Seagate Expansion Drive/bachelorProject/outputTest/"
g = glob(PATH+'/*.dcm')
#Id of the patient
id = 1
lowThreshold = 809
marchingStep = 1 #Step used for machring cubes algorithm, to create a 2d surface mesh from .dcm data
NR_PATIENTS = 4
#########################Functions####################################

#loads DICOM images from folder into list for manipulation
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

#converts raw values to Houndsvield units
def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)
    image[image == -2000] = 0
    #Convert raw values to Houndsfield units
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)
    return np.array(image, dtype=np.int16)


########################HU Histogram###################################
#Perform only once, costs quite a lot of time to read all slices
for i in range(1,   NR_PATIENTS+1):
    tempStr = 'pat'+str(i)
    patient = load_scan(PATH+tempStr)
    imgs = get_pixels_hu(patient)
     #save new images to data set to save reprocessing time
    np.save(O_PATH + "fullimages_%d.npy" % (i), imgs)

#create histogram of all voxel data
# for i in range(1, 16):

#     file_used = O_PATH+"fullimages_%d.npy" % i
#     imgs_to_process = np.load(file_used).astype(np.float64)

# plt.hist(imgs_to_process.flatten(), bins=50, color='c')
# plt.xlabel("Hounsfield Units (HU)")
# plt.ylabel("Frequency")
# plt.show()

###################Displaying an image stack###########################
imgs_to_process = np.load(O_PATH+"fullimages_%d.npy" % id)
patient = load_scan(PATH+'pat1')

def sample_stack(stack, rows=6, cols=6, start_with=10, show_every=6):
    fig, ax = plt.subplots(rows, cols, figsize=[12,12])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        ax[int(i/rows), int(i%rows)].set_title('slice %d' % ind)
        ax[int(i/rows), int(i%rows)].imshow(stack[ind], cmap='gray')
        ax[int(i/rows), int(i%rows)].axis('off')
    plt.show()

sample_stack(imgs_to_process)

##########################Resampling####################################
imgs_to_process = np.load(O_PATH+'fullimages_%d.npy' % id)
def resample(image, scan, new_spacing=[1,1,1]):
    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = np.array(list(spacing))

    resize_factor = spacing/new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

    return image, new_spacing

print("Shape before resampling\t", imgs_to_process.shape)
imgs_after_resamp, spacing = resample(imgs_to_process, patient, [1,1,1])
print("Shape after resampling\t", imgs_after_resamp.shape)


###################Creating a 3D-plot333333333##########################
def make_mesh(image, threshold=-300, step_size=1):
    print("Transposing Surface")
    p = image.transpose(2,1,0)
    print("Calculating Surface")
    verts, faces, norm, val = measure.marching_cubes(p, threshold, step_size=step_size, allow_degenerate=True)
    return verts, faces

def plotly_3d(verts, faces):
    x, y, z = zip(*verts)
    print("Drawing")
    colormap = ['rgb(1,0,0)', 'rgb(1,0,0)']
    fig = FF.create_trisurf(x=x, y=y, z=z, plot_edges=False, colormap='Portland', simplices=faces, backgroundcolor='rgb(64,64,64)', title="Interactive Visualization")
    py.iplot(fig, filename="testPlot")

def plt_3d(verts, faces):
    print("Drawing")
    x, y, z, = zip(*verts)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces], linewidths=0.05, alpha=1)
    face_color = [1, 1, 0.9]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, max(x))
    ax.set_ylim(0, max(y))
    ax.set_zlim(0, max(z))
    ax.set_facecolor((0.7, 0.7, 0.7))
    plt.show()

v, f = make_mesh(imgs_after_resamp, lowThreshold, marchingStep)
#print("Mesh constructed")
plotly_3d(v,f)
#plt_3d(v, f)