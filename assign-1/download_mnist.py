'''
Downloads MNIST Data into a path.
'''

# Standard libs
import os, struct
import gzip
import wget, shutil

import numpy as np

def load_mnist(path='data',refresh=False):
    """  
    Download MNIST from Le Cunn's page.
    See : 

    Args:

    * path : path to download to 
    * refresh: force redownload of data.

    Returns:
    DATA={'train': (images,labels),'fold_i':(images_i,labels_i)}
    """
    DATA={}
    if not os.path.exists(os.path.join(os.curdir, path)) or refresh:
        os.mkdir(os.path.join(os.curdir,path))
        wget.download('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', out=path)
        wget.download('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', out=path)
        wget.download('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', out=path)
        wget.download('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', out=path)

        source_f=['train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz','t10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz']
        dest_f=[s[:-3] for s in source_f]

        for source,dest in zip(source_f,dest_f):
            with gzip.open(os.path.join(path,source), 'rb') as f_in:
                with open(os.path.join(path,dest), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

    kind='train'
    labels_path = os.path.join(path, '{kind}-labels-idx1-ubyte'.format(kind=kind))
    images_path = os.path.join(path, '{kind}-images-idx3-ubyte'.format(kind=kind))
    
    with open(labels_path, 'rb') as lpath:
        magic, n = struct.unpack('>II', lpath.read(8))
        labels = np.fromfile(lpath, dtype=np.uint8)
        labels = [vectorized_result(label) for label in labels]
        labels= np.array(labels)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath,  dtype=np.uint8).reshape(len(labels), 784)

    DATA['train']=list(zip(images,labels))

    kind='t10k'
    labels_path = os.path.join( path, '{kind}-labels-idx1-ubyte'.format(kind=kind))
    images_path = os.path.join( path, '{kind}-images-idx3-ubyte'.format(kind=kind))

    with open(labels_path, 'rb') as lpath:
        magic, n = struct.unpack('>II', lpath.read(8))
        labels = np.fromfile(lpath, dtype=np.uint8)
        labels = [vectorized_result(label) for label in labels]
        labels= np.array(labels)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath,  dtype=np.uint8).reshape(len(labels), 784)

    # Split the test dataset into folds
    indices=np.arange(len(labels))
    np.random.shuffle(indices)
    indice_list=np.split(indices,5)

    for i in range(5):
        fold='fold-{i}'.format(i=i)    
        DATA[fold]=list(zip(images[indice_list[i]],labels[indice_list[i]]))
    
    return DATA

def vectorized_result(y):
    e = np.zeros((10, 1))
    e[y] = 1.0
    return e

def test():

    DATA=load_mnist()
    print(DATA)

if __name__=='__main__':
    test()
