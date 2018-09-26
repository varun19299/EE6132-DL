'''
Visulaise Weights and Activations

'''

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

# Module imports
import model_lib
import vis_lib

tfe = tf.contrib.eager

 # Soft placement
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tfe.enable_eager_execution(config=config)

model=model_lib.create_model()

checkpoint_dir='/tmp/tensorflow/mnist/checkpoints'

checkpoint = tf.train.Checkpoint(model=model)
# Restore variables on creation if a checkpoint exists.
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

print(model.summary())
print(model.weights)

layer_dict = dict([(layer.name, layer) for layer in model.layers])

print(layer_dict)

# Visualise activations
max_nfmap = np.Inf ## print ALL the images
input_img = model.layers[0].input
visualizer = vis_lib.VisualizeImageMaximizeFmap(model=model,
pic_shape = (96,96,1))
print("find images that maximize feature maps")
argimage = visualizer.find_images(input_img,
                                  ['conv2d','conv2d_1'],
                                  layer_dict, 
                                  max_nfmap)
print("plot them...")
visualizer.plot_images_wrapper(argimage,n_row = 8, scale = 1)