import numpy as np
from PIL import ImageDraw
import cv2
import numpy as np
from sklearn import svm

bin_n = 16  # Number of bins
svm_params = dict(kernel_type=cv2.SVM_LINEAR,
                  svm_type=cv2.SVM_C_SVC)

affine_flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR


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