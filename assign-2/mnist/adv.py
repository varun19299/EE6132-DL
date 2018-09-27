import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

# Module imports
import model_lib
tfe=tf.contrib.eager

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

x=np.random.randn((28,28,1))+128
LR=0.01
d=np.ones(*x.shape)
target=0

while (np.max(LR*d)> 1e-6):
    # Compute Gradients
    with tf.GradientTape() as tape:
        C = model(x,training=False)[target]

    d = tape.gradient(C,x)

    x+= LR *d
