import sys
import numpy as np
from PNM import *

def CreateAndSavePFM(out_path):
    width = 512
    height = 512
    numComponents = 3

    img_out = np.empty(shape=(width, height, numComponents), dtype=np.float32)

    for y in range(height):
        for x in range(width):
            img_out[y,x,:] = 1.0

    writePFM(out_path, img_out)

def LoadAndSavePPM(in_path, out_path):
    img_in = loadPPM(in_path)
    img_out = np.empty(shape=img_in.shape, dtype=img_in.dtype)
    height,width,_ = img_in.shape # Retrieve height and width
    for y in range(height):
        for x in range(width):
            img_out[y,x,:] = img_in[y,x,:] # Copy pixels

    writePPM(out_path, img_out)

def LoadAndSavePFM(in_path, out_path):
    img_in = loadPFM(in_path)
    img_out = np.empty(shape=img_in.shape, dtype=img_in.dtype)
    height,width,_ = img_in.shape # Retrieve height and width
    for y in range(height):
        for x in range(width):
            img_out[y,x,:] = img_in[y,x,:] # Copy pixels

    writePFM(out_path, img_out)

def LoadPPMAndSavePFM(in_path, out_path):
    img_in = loadPPM(in_path)
    img_out = np.empty(shape=img_in.shape, dtype=np.float32)
    height,width,_ = img_in.shape
    for y in range(height):
        for x in range(width):
            img_out[y,x,:] = img_in[y,x,:]/255.0

    writePFM(out_path, img_out)

def LoadPFMAndSavePPM(in_path, out_path):

    img_in = loadPFM(in_path)
    print (img_in.shape)
    img_out = np.empty(shape=img_in.shape, dtype=np.float32)
    height,width,_ = img_in.shape
    for y in range(height):
        for x in range(width):
            img_out[y,x,:] = img_in[y,x,:] * 255.0

    writePPM(out_path, img_out.astype(np.uint8))

########################################################################
def MapingLatlong(input_path, out_path):
    """
    :param input_path:
    :param out_path:
    :return:
    """
    latlong = loadPFM(input_path)
    print('shape of the latlong', latlong.shape)

    y_length, x_length, channel = latlong.shape

    # initialize the output image
    diameter = 511
    radius = float(diameter)/2
    v = [0.0, 0.0, 1.0] #viewing vector
    light_Probe = np.empty(shape=(diameter, diameter, channel), dtype=np.float32)
    print('shape of the light Prob', light_Probe.shape)

    # do the mapping for each pixels
    for h in range(0, diameter): # height of the image
        for w in range(0, diameter): # weight of the image
            dis = np.sqrt(np.power(h-radius, 2) + np.power(w-radius, 2))
            # distance between each pixel to the center
            if dis >= radius:
                for ch in range(3):
                    light_Probe[h][w][ch] = 0.0
            else:
                n = normVector(h,w,radius)
                # this n has been normalised, reflection vector
                r = (2*(np.dot(n, v))*n - v)
                # the Cartesian coordinates after conversion
                x, y = CartesianToSpherical(r, y_length, x_length)
                # mapping back the sphere
                for ch in range(channel):
                    if latlong[y][x][ch] > 1.0:
                        latlong[y][x][ch] = 1.0
                    light_Probe[h][w][ch] = latlong[y][x][ch]

    writePFM(out_path, light_Probe)

def normVector(height, weight, radius):
    """
    :param height:
    :param weight:
    :param radius:
    :return:
    """
    y = radius - height
    x = weight - radius
    z = np.sqrt(np.power(radius, 2) - np.power(y, 2) - np.power(x, 2))

    return np.array([x / radius, y / radius, z/radius])

def CartesianToSpherical(rVector, y_length, x_length):
    """
    :param rVector:
    :param y_length:
    :param x_length:
    :return:
    """
    # chang the angle phi from (-pi , pi) to (0, 2pi)
    x, y, z = rVector[0], rVector[1], rVector[2]
    phi = np.arctan2(x, z) + np.pi
    theta = np.arccos(y)
    # phi = np.arctan2(y, x) + np.pi
    # theta = np.arccos(z)

    a = int(((phi) / (2 * np.pi)) * x_length)
    b = int((theta / np.pi) * y_length)

    return a, b
import math
def Gamma(path_in, gamma, stop, path_out):
    """
    :param img_path:
    :param stop:
    :param gamma:
    :param out_path:
    :return:
    """
    img = loadPFM(path_in)
    height, width, channel = img.shape

    imageOut = np.empty(shape=img.shape, dtype=img.dtype)


    for h in range(height):
        for w in range(width):
            for ch in range(channel):
                img[h][w][ch] *= stop
                img[h][w][ch] = np.power(img[h][w][ch], 1/gamma)  # gamma correction
                if img[h][w][ch] > 1.0:
                    imageOut[h][w][ch] = 255
                else:
                    imageOut[h][w][ch] = math.ceil(img[h][w][ch]*255)
    writePPM(path_out, imageOut.astype(np.uint8))


import os
if '__main__' == __name__:
    pass
    # MapingLatlong('../UrbanProbe/urbanEM_latlong.pfm','../UrbanProbe/result2.pfm')
    # gamma = [1.8,2.0]
    # stop = [3,5,7,9]
    # for g in gamma:
    #     for s in stop:
    #         Gamma('../UrbanProbe/result2.pfm', g, s, '../UrbanProbe/result2AfterGamma'+str(g)+'Stop'+str(s)+'.pfm' )
    #
