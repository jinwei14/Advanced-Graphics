def MedianCutSampling(partitions):
    img_in = [[0,0,0],[0,0,0]]

    def change(img_in):
        img_in[0][0] = 1

    change(img_in)

    print img_in

MedianCutSampling(2)