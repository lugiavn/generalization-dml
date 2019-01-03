# Generalization in Metric Learning: Should the Embedding Layer be the Embedding Layer?

![ZZ](dml.png?raw=true "X")

This reproduces the result from our paper https://arxiv.org/abs/1803.03310

The code isn't too clean for now, but it should work


## Benchmarks

**R@1** retrieval performance

Dataset | Cars-196 | CUB-200-2011 | Stanford Online Product
------------ | ------------- | ------------- | -------------
Lifted structure [17] |  53.0 | 47.2 | 62.5
HDC [40] |  73.7 | 53.6 | 70.9
N-pair [27] | 71.1 | 51.0 | 67.7
Proxy-NCA [16] | 73.2 | 49.2 | 73.7
Our (layer **pool5.3**) |  **87.8** | **66.4** | **74.8**


## Prerequisite

Python, pytorch, numpy, scipy, etc.


## Produce the result on Cars-196

Download cars-196 dataset from http://ai.stanford.edu/~jkrause/cars/car_dataset.html (cars_annos.mat & unzip car_ims.tgz)

Edit train.py, update the value of dataset_path

Run the training: python train.py

To reproduce result on CUB and OnlineProduct dataset, refer to this https://github.com/lugiavn/generalization-dml/issues/1

## Notes

We used pytorch version '0.3.0.post4' and scipy '0.19.1'

We noticed scipy.misc.imresize behaviour could vary, so using a different version of scipy might result in slightly worse/different performance than what's reported in our paper.

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
