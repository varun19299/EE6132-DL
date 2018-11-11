# README

All codes are to be run from the base of the folder.  
` export PYTHONPATH="."`

## Dependencies

* Python > 3.5
* Tensorflow > 1.8
* Numpy, matplotlib, opencv > 3.3

## Training the mnist model

```
python3 mnist/mnist_eager.py
tensorboard --log_dir /tmp/tensorflow/mnist/
```

Logs including those of accuracy, losses and predictions on images show up on tensorboard.

## Questions 2,3

These can be viewed at the respective notebooks:

* `mnist/visualise.ipynb` for visualising model filters, activations and performing occlusion region tests.

* `mnist/adversarial.ipynb` for non-targeted (with and without source regularisation), targeted attacks and noise based attacks.