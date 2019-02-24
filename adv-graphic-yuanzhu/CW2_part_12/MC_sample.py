import sys
import numpy as np
import math

from PNM import *


######################################### CODE START FROM HERE #########################################

# exponential scale on HDR
def exp_scale(F, stop):

    height, width, channel = F.shape

    for h in range(height):
        for w in range(width):
            for ch in range(3):
                for t in range(stop):
                    F[h, w, ch] *= 2
                if F[h, w, ch] > 255:
                    F[h, w, ch] = 255

    print('finish exp tone...')

    return F


# gamma function on HDR
def gamma_scale(F, gam):

    height, width, channel = F.shape
    for h in range(height):
        for w in range(width):
            for ch in range(channel):
                F[h, w, ch] = ((F[h, w, ch]/255) ** (1. / gam))*255

    print('finish gamma tone...')

    return F


# sample from EM
def MC_sample(sample_num, show_sample_only):

    img = loadPFM('../GraceCathedral/grace_latlong.pfm')
    height, width, channel = img.shape
    F = np.zeros([height, width, channel])

    # make a copy
    for h in range(height):
        for w in range(width):
            for c in range(channel):
                F[h][w][c] = img[h][w][c]

    intensity = np.sum(F, axis=2)/3

    for h in range(height):
        for w in range(width):
            for ch in range(channel):
                if F[h, w, ch]>255:
                    F[h][w][ch] = 255
            intensity[h][w] *= math.sin((h/(height-1.)) * np.pi)


    cdf_1d = np.sum(intensity, axis=1)
    cdf_1d /= np.sum(cdf_1d)

    cdf_2d = np.zeros((height, width))
    for h in range(1,height):
        this_cdf_2d = intensity[h]/np.sum(intensity[h])
        cdf_1d[h] += cdf_1d[h-1]
        for w in range(1,width):
            this_cdf_2d[w] += this_cdf_2d[w-1]
        cdf_2d[h] = this_cdf_2d

    # print a map only contains samples
    if show_sample_only:
        sample_map = np.zeros([height, width, channel])


    # find coordinate of each sample on EM
    for sample in range(sample_num):

        this_i, this_j = -1, -1

        # sample by uniform random variate
        u_i = np.random.uniform(0, 1)
        for h in range(height):
            if cdf_1d[h] >= u_i:
                this_i = h
                u_j = np.random.uniform(0, 1)
                for w in range(width):
                    if cdf_2d[this_i][w] >= u_j:
                        this_j = w
                        break
                break

        # draw sample points and window
        for coord in [[this_i-2, this_j-2], [this_i-2, this_j-1], [this_i-2, this_j], [this_i-2, this_j+1],
                      [this_i+2, this_j-1], [this_i+2, this_j], [this_i+2, this_j+1], [this_i+2, this_j+2],
                      [this_i-2, this_j+2], [this_i-1, this_j+2], [this_i, this_j+2], [this_i+1, this_j+2],
                      [this_i-1, this_j-2], [this_i, this_j-2], [this_i+1, this_j-2], [this_i+2, this_j-2],
                      [this_i, this_j]]:
            if -1 < coord[0] < 512 and -1 < coord[1] < 1024:
                if show_sample_only:
                    sample_map[coord[0]][coord[1]] = img[this_i][this_j]

                F[coord[0]][coord[1]] = [0., 0., 80.]

    F = exp_scale(F, 6)
    F = gamma_scale(F, 2.2)
    writePPM('../MC_{}.ppm'.format(sample_num), F.astype(np.uint8))

    # writePFM('../this_sample.pfm', F)
    if show_sample_only:
        sample_map = exp_scale(sample_map, 6)
        sample_map = gamma_scale(sample_map, 2.2)
        writePPM('../SAMPLE_MAP_{}.ppm'.format(sample_num), sample_map.astype(np.uint8))


# MC samples
if '__main__' == __name__:

    MC_sample(256, True)
