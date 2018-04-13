from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
from matplotlib import cm
from numpy import array
import math
import stl
import numpy

def find_mins_maxs(obj):
    minx = maxx = miny = maxy = minz = maxz = None
    for p in obj.points:
        if minx is None:
            minx = p[stl.Dimension.X]
            maxx = p[stl.Dimension.X]
            miny = p[stl.Dimension.Y]
            maxy = p[stl.Dimension.Y]
            minz = p[stl.Dimension.Z]
            maxz = p[stl.Dimension.Z]
        else:
            maxx = max(p[stl.Dimension.X], maxx)
            minx = min(p[stl.Dimension.X], minx)
            maxy = max(p[stl.Dimension.Y], maxy)
            miny = min(p[stl.Dimension.Y], miny)
            maxz = max(p[stl.Dimension.Z], maxz)
            minz = min(p[stl.Dimension.Z], minz)
    return minx, maxx, miny, maxy, minz, maxz

def translate(_solid, step, padding, multiplier, axis):
    if axis == 'x':
        items = [0, 3, 6]
    elif axis == 'y':
        items = [1, 4, 7]
    elif axis == 'z':
        items = [2, 5, 8]
    for p in _solid.points:
        # point items are ((x, y, z), (x, y, z), (x, y, z))
        for i in range(3):
            p[items[i]] += (step * multiplier) + (padding * multiplier)


#create new plot
figure = pyplot.figure()
axes = mplot3d.Axes3D(figure)

#load .stl file
mesh = mesh.Mesh.from_file('/media/ruben/Seagate Expansion Drive/bachelorProject/test/Geit Gerda_Test_001.stl')


total = len(mesh.v0) + len(mesh.v1) + len(mesh.v2)
print('Total amount of vertices: ', total)

# #Put vertices in a file
# file = open('vertices.txt', 'w')
# for i in range(0,len(mesh.v0)):
#     triangle = str(mesh.v0[i]) + str(mesh.v1[i]) + str(mesh.v2[i]) + '\n'
#     file.write(str(triangle))
# file.close()

#Auto scale to mesh size

minx, maxx, miny, maxy, minz, maxz = find_mins_maxs(mesh)
w = -maxx
l = -maxy
h = -maxz

translate(mesh, w, 1, 1, 'x')
translate(mesh, l, 1, 1, 'y')
translate(mesh, h, 1, 1, 'z')

axes.add_collection3d(mplot3d.art3d.Poly3DCollection(mesh.vectors, facecolor=(1,0,0), edgecolor='b'))
scale = mesh.points.flatten(-1)
axes.auto_scale_xyz(scale, scale, scale)

axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_zlabel('z')

pyplot.show()
