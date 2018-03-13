# Generalization in Metric Learning: Should the Embedding Layer be the Embedding Layer?

This reproduces the result from our paper https://arxiv.org/abs/1803.03310

The code isn't too clean for now, but it should work

## Prerequisite

Python, pytorch, numpy, etc.


## Produce the result on Cars-196

Download & unzip cars-196 dataset from http://ai.stanford.edu/~jkrause/cars/car_dataset.html

Edit train.py, update the value of dataset_path

Run the training: python train.py

## 

```pyhon
>>> scipy.__version__
'0.19.1'
>>> scipy.misc.imresize(np.array([[0, 100, 200]]), [1,6])
array([[  0,  32,  95, 159, 223, 255]], dtype=uint8)
```

```pyhon
>>> scipy.__version__
'1.0.0'
>>> scipy.misc.imresize(np.array([[0, 100, 200]]), [1,6])
array([[  0,   0,  63, 127, 191, 255]], dtype=uint8)
```
