from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import imutils
from skimage import measure
import plotly.figure_factory as FF
import plotly.plotly as py
import scipy.ndimage
np.set_printoptions(threshold=np.nan)

#binary_slices_for_model = np.load('binary_tooth_slices.npy')
binary_slices_for_model = np.load('binary_expert_slices.npy')
print('shape binary tooth slices: ', np.shape(binary_slices_for_model))
resized_images = []

def largest_dims():
    H = W = 0
    for img in binary_slices_for_model:
        height, width = np.shape(img)
        if H < height:
            H = height
        if W < width:
            W = width
    return H, W


#Make sure the images are a square
def reshapeImgs(expert_model = False):
    if(expert_model):
        H, W = largest_dims()
        for idx, img in enumerate(binary_slices_for_model):
            #plt.imshow(image, cmap='jet')
            #plt.show()
            height, width = np.shape(img)
            difH = H-height
            difW = W-width
            #top, bottom, left, right
            resized_images.append(cv2.copyMakeBorder(img, 0, difH, 0, difW, cv2.BORDER_CONSTANT, value=0))
            print(np.shape(resized_images[-1]))
    else:
        for idx, img in enumerate(binary_slices_for_model):
            #plt.imshow(image, cmap='jet')
            #plt.show()

            difference = abs(min((len(img)-len(img[1])), (len(img[1])-len(img))))
            height, width = np.shape(img)
            if height > width:
                resized_images.append(cv2.copyMakeBorder(img, 0, 0, 0, difference, cv2.BORDER_CONSTANT, value=0))
            if height < width:
                resized_images.append(cv2.copyMakeBorder(img, 0, difference, 0, 0, cv2.BORDER_CONSTANT, value=0))
            print(np.shape(resized_images[idx]))
        print(np.shape(binary_slices_for_model))

def resample(image, new_spacing=[1,1,1]):
    image = scipy.ndimage.interpolation.zoom(image, 0.5)
    return image

def make_3d_array():
    D3_array = resized_images[0]
    for image in resized_images[1:]:
        D3_array = np.dstack((D3_array, image))
    return D3_array.transpose()

def construct3DModel():
    # Make 3D structure from 2D binary slices
    D3_array = resized_images[0]
    for image in resized_images[1:]:
        D3_array = np.dstack((D3_array, image))

    x = np.arange(D3_array.shape[0])[:, None, None]
    y = np.arange(D3_array.shape[1])[None, :, None]
    z = np.arange(D3_array.shape[2])[None, None, :]
    x, y, z = np.broadcast_arrays(x, y, z)
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    print(np.shape(x))
    c = np.tile(D3_array.ravel()[:, None], [1,3])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(x.ravel(), y.ravel(), z.ravel(), c=c)
    #plt.show()
    surf = ax.plot_trisurf(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def make_mesh(image, threshold = 0, step_size=1):
    print('Transposing Surface')
    p = image.transpose(2,1,0)
    print('Calculating Surface')
    verts, faces, norm, val = measure.marching_cubes(p, threshold, step_size=step_size, allow_degenerate=True)
    return verts, faces

def plotly_3d(verts, faces):
    x, y, z = zip(*verts)
    print('Drawing')
    colormap = ['rgb(1,0,0)', 'rgb(1,0,0)']
    fig = FF.create_trisurf(x=x, y=y, z=z, plot_edges=False, colormap='Portland', simplices=faces, backgroundcolor='rgb(64,64,64)', title="Interactive Visualization")
    py.iplot(fig, filename="tooth_test_2")
#construct3DModel()
reshapeImgs(expert_model = True)
d3_structure = make_3d_array()
print(np.shape(d3_structure))
d3_structure = resample(d3_structure)
v, f = make_mesh(d3_structure)
plotly_3d(v,f)