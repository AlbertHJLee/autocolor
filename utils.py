import numpy as np
import os
from glob import glob
from scipy import misc



def rgb2gray(image):
    
    # Convert RGB image to grayscale using luma https://en.wikipedia.org/wiki/Luma_(video)
    grayimg = 0.2126*image[:,:,0] + 0.7152*image[:,:,1] + 0.0722*image[:,:,2]
    grayimg = np.minimum(grayimg*0.+255.,grayimg)
    grayimg = np.maximum(grayimg*0.,grayimg)
    
    return grayimg



def pushwhite(image):

    # Push non-black colors to white
    lines = np.minimum(image*0.+255.,(image**3)*20.)

    return lines



def decolor():
    
    # get raw color files and convert to "line art"
    # read in files
    files = glob(os.path.join("raws","*.png"))
    rawimages = np.array([misc.imread(file) for file in files])
    imgindex = np.zeros(rawimages.shape[0],dtype=np.float32)
    
    # only use images sufficiently different from one another
    imgindex[0] = 10.
    for i in range(rawimages.shape[0]-1):
        imgindex[i+1] = np.mean( np.sqrt((rawimages[i,:,:,:]-rawimages[i+1,:,:,:])**2) )
    uniqueimgs = (imgindex > 1.5)
        # right now imgindex threshold needs to be tuned by hand
        # should find better way to do this
    
    # convert to grayscale and push grays to white
    decolored = np.zeros(rawimages.shape[:3])
    newshape = rawimages.shape[1:]
    print(rawimages.shape)
    #truth = np.zeros(rawimages.shape)
    for i in range(rawimages.shape[0]-1):
        if (uniqueimgs[i]):
            decolored[i,:,:] = rgb2gray(rawimages[i,:,:,:])

    return [rawimages,decolored,imgindex]

