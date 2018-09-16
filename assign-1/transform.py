import numpy as np
from PIL import ImageDraw
import cv2
import numpy as np
from sklearn import svm

bin_n = 16  # Number of bins

affine_flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
SZ = 28

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=affine_flags)
    return img

def noise_addition(DATA,sigma=1e-4):
    def _pertub(x,sigma):
        return x+np.random.randn(*x.shape)*sigma
    l=len(DATA)
    for i in range(l):
        x,y=DATA[i][0],DATA[l][1]
        if i%10==0:
            DATA.append(_pertub(x,sigma),y)
        if i%10==1:
            img=deskew(x.reshape((28,28))).reshape((784,1))
            DATA.append(img,y)
    print("New Dataset size ",len(DATA))
    return DATA

def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    # quantizing binvalues in (0...16)
    bins = np.int32(bin_n * ang / (2 * np.pi))
    bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
    mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n)
             for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist

def preprocess(DATA, rows=28, cols=28):
    DATA2=[]
    for x,y in DATA:
        hogdata = hog(deskew(x.reshape(rows, cols))) 
        x_=np.float32(hogdata).reshape(64,)
        DATA2.append([x_,y])
    return DATA2