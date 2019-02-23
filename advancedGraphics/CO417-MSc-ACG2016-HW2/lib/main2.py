import sys
import numpy as np
from matplotlib import pyplot as mp
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

    writePFM(out_path, img)

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
    img_out = np.empty(shape=img_in.shape, dtype=np.float32)
    height,width,_ = img_in.shape
    for y in range(height):
        for x in range(width):
            img_out[y,x,:] = img_in[y,x,:] * 255.0

    writePPM(out_path, img_out.astype(np.uint8))

def LoadPFM(in_path):
    img_in = loadPFM(in_path)
    height,width,_ = img_in.shape # Retrieve height and width
    return img_in

def Gaussian(x):
    return np.exp(-np.power(x - 0.5, 2.) / (2 * np.power(0.1, 2.))) # mu = 0.5, sig = 0.1

def AssembleHDR(img_out):
    Z = []
    Z.append(LoadPFM("../Memorial/memorial1.pfm"))
    Z.append(LoadPFM("../Memorial/memorial2.pfm"))
    Z.append(LoadPFM("../Memorial/memorial3.pfm"))
    Z.append(LoadPFM("../Memorial/memorial4.pfm"))
    Z.append(LoadPFM("../Memorial/memorial5.pfm"))
    Z.append(LoadPFM("../Memorial/memorial6.pfm"))
    Z.append(LoadPFM("../Memorial/memorial7.pfm"))

    d_ti = [1.0, 4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0]

    height, width, _ = Z[0].shape
    F = np.empty(shape=Z[0].shape, dtype=Z[0].dtype)

    intensity = 0.0
    up_ = np.zeros(3)
    bot_ = 0.0
    for i in range(height):
        for j in range(width):
            for x in range(7):
                intensity = (Z[x][i][j][0]+Z[x][i][j][1]+Z[x][i][j][2])/3
                if (intensity > 0.005) and (intensity < 0.92):
                    for y in range(3):
                        up_[y] += np.log((1.0/d_ti[x])*Z[x][i][j][y])*Gaussian(intensity)
                    bot_ += Gaussian(intensity)
            if not bot_ == 0.0:
                F[i][j][0] = np.exp(up_[0]/bot_)
                F[i][j][1] = np.exp(up_[1]/bot_)
                F[i][j][2] = np.exp(up_[2]/bot_)
            else:
                F[i][j][0] = 0.0
                F[i][j][1] = 0.0
                F[i][j][2] = 0.0
            up_ = np.zeros(3)
            bot_ = 0.0

    writePFM(img_out, F)

def LinearToneMapper(img_path, scaling, out_path):
    img_in = LoadPFM(img_path)
    height, width, _ = img_in.shape

    brightest = 0.0
    dimmest = 1.0

    N = np.empty(shape=img_in.shape, dtype=img_in.dtype)

    for i in range(height):
        for j in range(width):
            intensity = (img_in[i][j][0]+img_in[i][j][1]+img_in[i][j][2])/3
            if intensity > brightest: brightest = intensity
            if intensity < dimmest and not intensity == 0.0: dimmest = intensity
            for x in range(3):
                img_in[i][j][x] *= scaling  # scaling factor best = 1024.
                if img_in[i][j][x] > 1.0:
                    N[i][j][x] = 255
                else:
                    N[i][j][x] = math.ceil(img_in[i][j][x]*255)

    print 'Brightest:'+brightest+'Dimmest:'+ dimmest
    writePPM(out_path, N.astype(np.uint8))

def GammaCorrection(img_path, stop, gamma, out_path):
    img_in = LoadPFM(img_path)
    height, width, _ = img_in.shape

    G = np.empty(shape=img_in.shape, dtype=img_in.dtype)

    for i in range(height):
        for j in range(width):
            # print img_in[i][j][0],img_in[i][j][1],img_in[i][j][2]
            for x in range(3):
                img_in[i][j][x] *= stop # scaling factor
                img_in[i][j][x] = pow(img_in[i][j][x], 1./gamma)  # gamma correction
                if img_in[i][j][x] > 1.0:
                    G[i][j][x] = 255
                else:
                    G[i][j][x] = math.ceil(img_in[i][j][x]*255)
    writePPM(out_path, G.astype(np.uint8))

def magnitude(v):
    return math.sqrt(sum(v[i]*v[i] for i in range(len(v))))

def sub(u, v):
    return [u[i]-v[i] for i in range(len(u))]

def add(u, v):
    return [u[i]+v[i] for i in range(len(u))]

def mult(u, v):
    return [u[i]*v for i in range(len(u))]

def divide(u, v):
    return [u[i]/v for i in range(len(u))]

def normalize(v):
    vmag = magnitude(v)
    return [v[i]/vmag for i in range(len(v))]

def dot(u, v):
    return np.sum(u[i]*v[i] for i in range(len(u)))

def normalize2(v):
    return np.array([v[i]/np.sum(v) for i in range(len(v))])

def Latlong(out_path):
    latlongMap = loadPFM("../GraceCathedral/grace_latlong.pfm")
    u_ = 512
    v_ = 256
    v = [0., 0., 1.] # viewing position
    radius = 255.5
    lightProbe = np.empty(shape=(511, 511, 3), dtype=np.float32) # output image initialization

    for i in range(0, 511):
        for j in range(0, 511):
            dis = math.sqrt(pow(i-radius, 2)+pow(j-radius, 2)) # calculate the distance between origin and current pixel
            if dis >= radius:
                for x in range(3):
                    lightProbe[i][j][x] = 0.0 # set pixels outside circle to black color 0,0,0
            else:
                print i,j
                dy = (radius-i)/radius
                dx = (j-radius)/radius
                dz = math.sqrt(1.0 - math.pow(dy, 2) - math.pow(dx, 2))

                if dz == 0:
                    n = [dx, dy, 0]
                else:
                    angle_az = math.atan2(dx, dz)
                    angle_po = math.acos(dy)

                    n = normalize([math.sin(angle_az)*math.sin(angle_po), math.cos(angle_po), math.cos(angle_az)*math.sin(angle_po)])
                r = normalize(sub(mult(n, 2.0*dot(n, v)), v))

                angle_az_r = math.atan2(r[0], r[2])+math.pi
                angle_po_r = math.acos(r[1])

                u_ = math.ceil(((angle_az_r)/(2*math.pi))*1023)
                v_ = math.ceil((angle_po_r/math.pi)*511)

                for x in range(3):
                    # temp = math.ceil(((n[x]+1)/2)*255)
                    # if temp > 255:
                    #    lightProbe[i][j][x] = 255
                    #else:
                    #    lightProbe[i][j][x] = temp
                    ttt = latlongMap[v_][u_][x]
                    if ttt > 1.0: ttt = 1.0
                    lightProbe[i][j][x] = ttt

    writePFM(out_path, lightProbe)


def Sampling(noOfSample):
    img_in = loadPFM("../GraceCathedral/grace_latlong.pfm")
    height,width,_ = img_in.shape  # 512 1025

    F = np.empty(shape=img_in.shape, dtype=img_in.dtype)

    for y in range(height):
        for x in range(width):
            F[y,x,:] = img_in[y,x,:]  # Copy pixels

    intensity = np.empty([height, width])
    pdf_sum_i = 0.0
    index_ij_p = []
    
    pdf_i = np.empty(height)
    cdf_i = np.empty(height)
    
    for i in range(height):
        for j in range(width):
            intensity[i][j] = (img_in[i, j, 0]+img_in[i, j, 1]+img_in[i, j, 2])/3.0*math.sin(i/511.0*np.pi) # Sum up the intensity of all pixels
            pdf_i[i] += intensity[i][j]
        pdf_sum_i += pdf_i[i]  # sum of pdf_i

    sum_intensity = np.sum(intensity)*(2.0*np.pi)/511/1023/noOfSample

    for n in range(height):
        pdf_i[n] /= pdf_sum_i  # normalize pdf_i
        if n > 0:
            cdf_i[n] = cdf_i[n-1]+pdf_i[n] # cumulative density function for each row i
        else:
            cdf_i[n] = pdf_i[n]
    
    for h in range(noOfSample):
        index_i = -1
        index_j = -1
        index_ij = []
        
        pdf_sum_j = 0.0
        cdf_j = np.empty(width)

        r = np.random.uniform(0,1) # random sample in range [0,1] = mu_1
        for x in range(height):
            if cdf_i[x] >= r:
                index_i = x
                break
        
        pdf_j = intensity[index_i] # get pdf_j
        
        for pd in pdf_j:
            pdf_sum_j += pd
            
        for k in range(width):
            pdf_j[k] /= pdf_sum_j # normalize pdf_j
            if k > 0:
                cdf_j[k] = cdf_j[k-1]+pdf_j[k]
            else:
                cdf_j[k] = pdf_j[k]

        r = np.random.uniform(0,1) # mu_2
        for y in range(width):
            if cdf_j[y] >= r:
                index_j = y
                break

        index_ij_p.append([index_i, index_j])
        
        for i in range(index_i-2, index_i+3):
            index_ij.append([i, index_j-2])
            index_ij.append([i, index_j+2])
                
        for j in range(index_j-2, index_j+3):
            index_ij.append([index_i-2, j])
            index_ij.append([index_i+2, j])
        
        F[index_i][index_j] = [0.0, 1.0, 0.0] # set sampling points to green
        for item in index_ij:
            if item[0] > -1 and item[0] < 512 and item[1] > -1 and item[1] < 1024:
                F[item[0]][item[1]] = [0.0, 1.0, 0.0] # draw 5X5 square around the sampling pixel.
    writePFM("part2_"+str(noOfSample)+".pfm", F)
    # GammaCorrection("part2_"+str(noOfSample)+".pfm", 1., 2.2, "part2_"+str(noOfSample)+".ppm")
    return noOfSample,index_ij_p,sum_intensity
# begin part 3:
def GenerateDiffuseBall(noOfSample, index_ij_p, sum_intensity):
    radius = 255.5
    img_in = loadPFM("../GraceCathedral/grace_latlong.pfm")
    height,width,_ = img_in.shape  # 512 1025
    lightProbe = np.empty(shape=(511, 511, 3), dtype=np.float32) # output image initialization

    for i in range(0, 511):
        for j in range(0, 511):
            dy = (radius-i)/radius
            dx = (j-radius)/radius
            dis = math.sqrt(pow(dy, 2)+pow(dx, 2)) # calculate the distance between origin and current pixel
            if dis < 1.0:
                dz = math.sqrt(1.0 - math.pow(dy, 2) - math.pow(dx, 2))
                n = np.array([dx, dy, dz])
                for s in range(noOfSample):
                    index_i = index_ij_p[s][0]
                    index_j = index_ij_p[s][1]

                    po_n = (index_i/511.0)*np.pi # polar angle
                    az_n = (index_j/1023.0)*np.pi*2 # azimuthal angle

                    n_x = -np.sin(az_n)*np.sin(po_n)
                    n_y = np.cos(po_n)
                    n_z = -np.cos(az_n)*np.sin(po_n)

                    cos_theta = np.dot(n, np.array([n_x, n_y, n_z]))
                    if cos_theta > 0:
                        lightProbe[i, j] += normalize2(img_in[index_i, index_j])*float(cos_theta) # sum up normalized colour which times with cos_theta
                lightProbe[i, j, :] *= sum_intensity
            else:
                lightProbe[i, j, :] = np.array([0.,0.,0.])

    writePFM("part3_"+str(noOfSample)+".pfm", lightProbe)
    # GammaCorrection("part3_"+str(noOfSample)+".pfm", 1., 2.2, "part3_"+str(noOfSample)+".ppm")
    
if '__main__' == __name__:
    # print "Sampling started! (64 samples)"
    # noOfSample,index_ij_p,sum_intensity = Sampling(64)
    # print "Sampling finished! (64 samples) Generating Diffuse Ball"
    # GenerateDiffuseBall(noOfSample,index_ij_p,sum_intensity)
    # print "Sampling started! (256 samples)"
    # noOfSample,index_ij_p,sum_intensity = Sampling(256)
    # print "Sampling finished! (256 samples) Generating Diffuse Ball"
    # GenerateDiffuseBall(noOfSample,index_ij_p,sum_intensity)
    # print "Sampling started! (1024 samples)"
    # noOfSample,index_ij_p,sum_intensity = Sampling(1024)
    # print "Sampling finished! (1024 samples) Generating Diffuse Ball"
    # GenerateDiffuseBall(noOfSample,index_ij_p,sum_intensity)
    # print "Finished!"

    pass
