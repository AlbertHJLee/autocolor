import numpy as np
import os
from glob import glob
from scipy import misc



def rgb2gray(image,option=0):
    
    # Convert RGB image to grayscale using luma https://en.wikipedia.org/wiki/Luma_(video)
    if option is 0:
        grayimg = 0.2126*image[:,:,0] + 0.7152*image[:,:,1] + 0.0722*image[:,:,2]
        grayimg = np.minimum(grayimg*0.+255.,grayimg)
        grayimg = np.maximum(grayimg*0.,grayimg)
    # Convert by fixing to true gray
    elif option is 1:
        grayimg = np.mean(image,axis=2)
    else:
        grayimg = np.mean(image,axis=2)
    
    return grayimg



def pushwhite(image):

    # Push non-black colors to white
    lines = np.minimum(((img/60.)**8)*30.,img*0.+255.)  # 25 is harder cut, 60 is safe

    return lines



def convolve(img1,img2):

    # Convolve two 2D image numpy arrays
    # img2 is considered the kernel, and we get the output after applying it as a filter on img1

    # Get image info and initialize constants
    kernelx = img2.shape[0]
    kernely = img2.shape[1]
    imagex = img1.shape[0]
    imagey = img1.shape[1]

    # Kernel must have odd dimensions
    if (kernelx % 2 == 0) or (kernely % 2 == 0):
        print('kernel must have odd dimensions')
        return img1

    output = img1*0.
    
    for i in range(kernelx):
        for j in range(kernely):
            if (i < (kernelx*.5) ):
                ia = int(kernelx*.5) - i
                ib = imagex
            else:
                ia = 0
                ib = imagex + int(kernelx*.5) - i
            if (j < (kernely*.5) ):
                ja = int(kernely*.5) - j
                jb = imagey
            else:
                ja = 0
                jb = imagey + int(kernely*.5) - j
            dx = i - int(kernelx*.5)
            dy = j - int(kernely*.5)
            output[ia:ib,ja:jb] = output[ia:ib,ja:jb] + \
                img2[i,j] * img1[(ia+dx):(ib+dx),(ja+dy):(jb+dy)]

    return output



    
def canny(image):

    # Canny edge detector https://en.wikipedia.org/wiki/Canny_edge_detector
    # http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html

    Dx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    Dy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    Gx = convolve(image,Dx)
    Gy = convolve(image,Dy)
    G = np.sqrt(Gx**2 + Gy**2)
    theta = np.arctan(Gy/(Gx+1e-20))

    return [G,theta]




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
      # the above is memory intensive - need workaround
    for i in range(rawimages.shape[0]-1):
        if (uniqueimgs[i]):
            decolored[i,:,:] = rgb2gray(rawimages[i,:,:,:])

    return [rawimages,decolored,imgindex]

