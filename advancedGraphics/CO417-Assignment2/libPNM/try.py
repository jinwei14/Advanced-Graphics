def MedianCutSampling(partitions):
    img_in = [[0,0,0],[0,0,0]]

    def change(img_in):
        img_in[0][0] = 1

    change(img_in)

    print img_in

MedianCutSampling(2)


print(int(1.5))

import numpy as np
tryarr = np.array([[1,2,3],[1,2,3]])

print (tryarr/float(3))*255

print (np.sum(tryarr))