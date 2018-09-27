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

x=np.random.randn(28,28,1)+128
print(x.shape)
LR=0.01
d=np.ones(x.shape)
target=0

x=tfe.Variable(x, dtype=tf.float32)
x= tf.reshape(x,(1,1,784))

path='/tmp/tensorflow/mnist/adv/non/'
count=0

while count < 100:
#while (np.max(LR*d)> 1e-3):
    # Compute Gradients
    with tf.GradientTape() as tape:
        tape.watch(x)
        C = model(x,training=False)[target]

    d = tape.gradient(C,x)
    print(d)
    x+= LR *d
    count+=1

plt.plot(x.numpy().reshape(28,28,1),cmap='grey')
plt.savefig(os.path.join(path,"adv0.jpg"))
plt.close()
