"""
@author:    Lafith Mattara
@date:      22-06-2022
"""

import utils_blindsr as blindsr
import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils_image as util

def run_degradation(
    path, scale=4,
    patch=72,
    savefig=False, showfig=True
    ):
    '''
    path    : full path to image
    scale   : scale factor for dowsampling
    patch   : patch size of LR
    savefig : saves LR-HR as a single image
    showfig : show LR-HR as a single image
    '''
    print("Reading image...")
    img = util.imread_uint(path, 3)
    img = util.uint2single(img)
    print("Running Degradation model...")
    lr, hr = blindsr.degradation_bsrgan(
        img, sf=scale, lq_patchsize=patch
        )
    print(
        "Input Image : ", img.shape,
        "LR :", lr.shape,
        "HR : ", hr.shape
        )
    print("Done!")
    if savefig or showfig:
        lr_nearest =  cv2.resize(
            util.single2uint(lr),
            (int(scale*lr.shape[1]),
            int(scale*lr.shape[0])),
            interpolation=0)
        img_concat = np.concatenate([lr_nearest, util.single2uint(hr)], axis=1)
        if savefig:
            util.imsave(img_concat, 'output.png')
        if showfig:
            plt.imshow(img_concat)
            plt.show()
    return lr, hr



if __name__ == "__main__":
    lr, hr = run_degradation('lenna.png', scale=4, patch=128)