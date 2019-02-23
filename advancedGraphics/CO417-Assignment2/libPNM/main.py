import sys
import numpy as np
from PNM import *
import math

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

import sys
def MedianCutSampling(partitions):


    def cut(startRow,startCol,endRow,endCol,itr,intensity,copy):
        # image = img_in[startRow:endRow+1,startCol:endCol+1]
        if itr == partitions:
            return

        height = endRow - startRow + 1
        width = endCol - startCol + 1

        maxDiff = sys.maxsize
        if height >= width: # this will cut off the height/row
            index = 0 # which row will be cut off
            for h in range(startRow,endRow+1):
                up, down= intensity[startRow:h+1,startCol:endCol+1],intensity[h:endRow+1,startCol:endCol+1]
                if np.sum(up) == np.sum(down):
                    index = h
                    break
                elif abs(np.sum(up)- np.sum(down))<maxDiff:
                    maxDiff = abs(np.sum(up)- np.sum(down))
                    index = h
            # print ('###################')
            # print (copy.shape)
            # print (copy[startRow:endRow + 1, startCol:endCol + 1].shape)
            # print (startRow)
            # print (startCol)
            # print (endRow)
            # print (endCol)
            # print (index)
            # print ('###################')
            if itr==partitions-1:
                index_list.append((int(0.5 * (startRow + index)),int(0.5*(startCol+endCol))))
                index_list.append((int(0.5 * (index + endRow)), int(0.5 * (startCol + endCol))))
            for w in range(width):
                copy[startRow:endRow+1,startCol:endCol+1][index-startRow][w] = [255.0, 255.0, 255.0] # set sampling points to green
            cut(startRow, startCol, index, endCol, itr+1, intensity, copy)
            cut(index, startCol, endRow, endCol, itr+1, intensity, copy)

        else: # this will cut off the width
            index = 0 # which column will be cut off
            for w in range(startCol,endCol+1):
                left, right = intensity[startRow:endRow+1,startCol:w+1], intensity[startRow:endRow+1,w:endCol+1]

                if np.sum(left) == np.sum(right):
                    index = w
                    break
                elif abs(np.sum(left) - np.sum(right)) < maxDiff:
                    maxDiff = abs(np.sum(left) - np.sum(right))
                    index = w
            # print ('###################')
            # print (copy.shape)
            # print (copy[startRow:endRow + 1, startCol:endCol + 1].shape)
            # print (startRow)
            # print (startCol)
            # print (endRow)
            # print (endCol)
            # print (index)
            # print ('###################')
            if itr == partitions - 1:
                index_list.append((int(0.5*(startRow+endRow)),int(0.5*(startCol+index))))
                index_list.append((int(0.5 * (startRow + endRow)), int(0.5 * (index + endCol))))

            for h in range(height):
                copy[startRow:endRow+1,startCol:endCol+1][h][index-startCol] = [255.0, 255.0, 255.0] # se
            cut(startRow, startCol, endRow, index, itr+1, intensity, copy)
            cut(startRow, index, endRow, endCol, itr+1, intensity, copy)


#########################
    img_in = loadPFM("../GraceCathedral/grace_latlong.pfm")  # 512 1024 3

    print('the shape of the image is',img_in.shape)

    height, width, _ = img_in.shape  # 512 1024
    #
    # copy = np.empty(shape=img_in.shape, dtype=img_in.dtype)
    #
    # for y in range(height):
    #     for x in range(width):
    #         copy[y, x, :] = img_in[y, x, :]  # Copy pixels

    intensity = np.empty([height, width])

    index_list = []
    for i in range(height):
        # temp = 0
        for j in range(width):
            intensity[i][j] = (img_in[i, j, 0]+img_in[i, j, 1]+img_in[i, j, 2])/3.0*np.sin((i/511.0)*np.pi) # Sum up the intensity of all pixels

    cut(0,0,height-1,width-1,0,intensity,img_in)

   # set a 5*5 grid
    for item in index_list:
        print (item)
        i,j = item[0],item[1]
        img_in[i][j] = [0.0, 0.0, 1.0]
        for row in range(i-2,i+3):
            for col in range(j-2,j+3):
                if (row == i-2 or row ==i+2 or col ==j-2 or col == j + 2 ):

                    img_in[row][col] = [0.0, 0.0, 1.0]




    writePFM('../GraceCathedral/part3_partisons'+str(np.power(2,partitions))+'.pfm', img_in)


if '__main__' == __name__:
    for i in [1,2,3,4,5,6,7,8]:
        print ('partisons' +str(np.power(2,i)))
        MedianCutSampling(i)




