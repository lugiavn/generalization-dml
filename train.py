import time
import os
import sys

import numpy as np
import scipy.io
import scipy.misc

import torch
import torchvision

from nams import *

NAME = 'cars.triplet.fc8.v2'
DESCRIPTION = ''

batch_size = 32
test_frequency = 1000
loss_frequency = 100

###############################################################
###############################################################
###############################################################

dataset_path = '.'
cars_annos = scipy.io.loadmat('./cars_annos.mat')

train_imgs = []
test_imgs = []
train_labels = []
test_labels = []
for d in cars_annos['annotations'][0]:
    im_path = d[0][0]
    bbox_x1 = d[1][0][0]
    bbox_y1 = d[2][0][0]
    bbox_x2 = d[3][0][0]
    bbox_y2 = d[4][0][0]
    imclass = d[5][0][0]
    test = d[6][0][0]
    test = imclass > 98
    
    if test:
        test_imgs.append(dataset_path + '/' + im_path)
        test_labels.append(imclass)
    else:
        train_imgs.append(dataset_path + '/' + im_path)
        train_labels.append(imclass)
        
print 'Done reading data'
print np.min(train_labels)
print np.max(test_labels)
print str(len(train_imgs)) + ' images for training'
print str(len(test_imgs)) + ' images for testing'


train_class2ids = []
while len(train_class2ids) <= np.max(train_labels):
    train_class2ids.append([])
for i in range(len(train_labels)):
    train_class2ids[train_labels[i]].append(i)

class_probs = []
for i in range(len(train_class2ids)):
    n = len(train_class2ids[i])
    if n == 1:
        n = 0
    class_probs.append(0.0 + n)
class_probs = np.array(class_probs) / np.sum(class_probs)


def construct_minibatch(all_imgs, all_labels = None, batch_size = None, indices = None, random_crop = False):
    if batch_size == None:
        batch_size = len(indices)
    data = np.zeros([batch_size, 3, 224, 224])
    if all_labels is not None:
        labels = np.zeros([batch_size])
    for i in range(batch_size):
        j = np.random.randint(low=0, high=len(all_imgs))
        if indices is not None:
            j = indices[i]
            
        # set label
        if all_labels is not None:
            labels[i] = all_labels[j]
            
        # read image
        img = scipy.misc.imread(all_imgs[j])
        if len(img.shape) < 3:
            img1 = img
            img = np.zeros((img.shape[0], img.shape[1], 3))
            img[:,:,0] = img1
            img[:,:,1] = img1
            img[:,:,2] = img1
            
        # resize
        if random_crop:
            s = max(224.0/img.shape[0], 224.0/img.shape[1])
            s *= 1.0 + np.random.rand() * 0.1
            img = scipy.misc.imresize(img, s)
            if np.random.rand() > 0.5:
                img = np.fliplr(img)
            img = my_random_crop(img, (224, 224))
        else:
            img = my_resize_image(img, (224, 224), mode=1)
            
        #data[i,:,:,:] = img.transpose([2, 0, 1]) / 255.0 - 0.5
        #data[i,:,:,:] = (img.transpose([2, 0, 1]) / 255.0 - 0.45) / 0.22
        img = img / 255.0
        img[:,:,0] = (img[:,:,0] - 0.485) / 0.229
        img[:,:,1] = (img[:,:,1] - 0.456) / 0.224
        img[:,:,2] = (img[:,:,2] - 0.406) / 0.225
        data[i,:,:,:] = img.transpose([2, 0, 1])
        
    if all_labels is not None:
        return data, labels
    else:
        return data

###############################################################
###############################################################
###############################################################

'''
x = torchvision.models.vgg16_bn(pretrained=False)
x = x.features
x2 = torchvision.models.vgg16_bn(pretrained=False)
x2 = x2.features
x3 = []
for i in range(40):
  x3.append(x[i])
x3.append(x2[40])
x3.append(x2[41])
x3.append(x2[42])
x3.append(x2[43])
x3 = torch.nn.Sequential(*x3)
'''

net_feature = torchvision.models.vgg16_bn(pretrained=True)
net_feature.features = torch.nn.Sequential(
    net_feature.features,
    torch.nn.MaxPool2d(7)
)
net_feature.classifier = torch.nn.Sequential(
    torch.nn.Linear(512, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 512)
)
myloss = MyTripletLoss(normalize_scale=4.0, learn_scale=True, DBL=True)

net_feature = net_feature.cuda()
myloss = myloss.cuda()

optimizer = torch.optim.SGD(
    [
        {'params': net_feature.features.parameters(), 'lr': 0.01, 'weight_decay': 5e-4},
        {'params': net_feature.classifier.parameters(), 'lr': 0.01, 'weight_decay': 5e-4},
        {'params': myloss.parameters(), 'lr': 0.01, 'weight_decay': 5e-5}
    ],
    lr=0.01, momentum=0.9, weight_decay=5e-4
)

nams_logger = NamLogger(str(time.time()) + '.' + NAME + '.namslog')
nams_logger.log('description', DESCRIPTION)

train_losses = []
rank1recalls = []
it = 0





###############################################################
###############################################################
###############################################################

def vgg_forward_all(input_var, vgg_model, out_layer = None):
    
    assert(len(net_feature.features[0]) == 44)
    assert(len(net_feature.classifier) <= 5)
    
    data = []
    x = input_var
    
    # get pool4
    for i in range(40):
        x = vgg_model.features[0][i](x)
    y = vgg_model.features[0][43](x)
    y = vgg_model.features[1](y)
    y = y.view(x.size(0), -1)
    data.append(y.data.cpu().numpy())
    
    # get pool5
    x = vgg_model.features[0][40](x)
    x = vgg_model.features[0][41](x)
    x = vgg_model.features[0][42](x)
    x = vgg_model.features[0][43](x)
    x = vgg_model.features[1](x)
    x = x.view(x.size(0), -1)
    data.append(x.data.cpu().numpy())
    
    # get fc6
    if len(net_feature.classifier) >= 1:
      x = vgg_model.classifier[0](x)
      data.append(x.data.cpu().numpy())
    
    # get fc7
    if len(net_feature.classifier) >= 3:
        x = vgg_model.classifier[1](x)
        x = vgg_model.classifier[2](x)
        data.append(x.data.cpu().numpy())
    
    # get fc8
    if len(net_feature.classifier) >= 5:
       x = vgg_model.classifier[3](x)
       x = vgg_model.classifier[4](x)
       data.append(x.data.cpu().numpy())

    return data

def perform_all_features_extraction(net, all_imgs, do_normalization=True):
    all_features = [[]]
    while len(all_features[0]) < len(all_imgs):
        indices = np.arange(len(all_features[0]), min(len(all_features[0])+batch_size, len(all_imgs)))
        data = construct_minibatch(all_imgs, indices=indices, random_crop=False)
        data = torch.autograd.Variable(torch.from_numpy(data).float().cuda())
        output = vgg_forward_all(data, net)
        for i in range(len(output)):
            try:
                all_features[i];
            except:
                all_features.append([])
            
            for j in range(output[i].shape[0]):
                all_features[i].append(output[i][j,:])
    print len(all_features[0])
    if do_normalization:
      for i in range(len(all_features)):
        all_features[i] = np.array(all_features[i])
        for j in range(all_features[i].shape[0]):
            all_features[i][j,:] = 100.0 * all_features[i][j,:] / np.linalg.norm(all_features[i][j,:])
    return all_features

def test_r1_full_all():
    imgs = test_imgs[::1]
    theirlabels = test_labels[::1]
    
    print 'start'
    net_feature.train(mode=False)
    all_features = perform_all_features_extraction(
        net_feature,
        imgs
    )
    print 'done'
    r1 = test_retrieval_r1(all_features[0], theirlabels)
    print 'R@1-full-pool4:', r1
    nams_logger.log('R@1-full-pool4', r1, step=it)
    
    r1 = test_retrieval_r1(all_features[1], theirlabels)
    print 'R@1-full-pool5:', r1
    nams_logger.log('R@1-full-pool5', r1, step=it)

    if len(all_features) >= 3:    
      r1 = test_retrieval_r1(all_features[2], theirlabels)
      print 'R@1-full-fc6:', r1
      nams_logger.log('R@1-full-fc6', r1, step=it)
    
    if len(all_features) >= 4:
        r1 = test_retrieval_r1(all_features[3], theirlabels)
        print 'R@1-full-fc7:', r1
        nams_logger.log('R@1-full-fc7', r1, step=it)
    
    if len(all_features) >= 5:
        r1 = test_retrieval_r1(all_features[4], theirlabels)
        print 'R@1-full-fc8:', r1
        nams_logger.log('R@1-full-fc8', r1, step=it)
    
    
    imgs = train_imgs[::1]
    theirlabels = train_labels[::1]

    net_feature.train(mode=False)
    all_features = perform_all_features_extraction(
        net_feature,
        imgs
    )
    r1 = test_retrieval_r1(all_features[0], theirlabels)
    print 'R@1-full-train-pool4:', r1
    nams_logger.log('R@1-full-train-pool4', r1, step=it)
    
    r1 = test_retrieval_r1(all_features[1], theirlabels)
    print 'R@1-full-train-pool5:', r1
    nams_logger.log('R@1-full-train-pool5', r1, step=it)
    
    if len(all_features) >= 3:
      r1 = test_retrieval_r1(all_features[2], theirlabels)
      print 'R@1-full-train-fc6:', r1
      nams_logger.log('R@1-full-train-fc6', r1, step=it)
    
    if len(all_features) >= 4:
        r1 = test_retrieval_r1(all_features[3], theirlabels)
        print 'R@1-full-train-fc7:', r1
        nams_logger.log('R@1-full-train-fc7', r1, step=it)
    
    if len(all_features) >= 5:
        r1 = test_retrieval_r1(all_features[4], theirlabels)
        print 'R@1-full-train-fc8', r1
        nams_logger.log('R@1-full-train-fc8', r1, step=it)
    
    nams_logger.flush()


###############################################################
###############################################################
###############################################################



tic = time.time()
for it in range(it, 96666):
    
    if it < 100:
        optimizer.param_groups[0]['lr'] = 0.00
        optimizer.param_groups[1]['lr'] = 0.01
        optimizer.param_groups[2]['lr'] = 0.01
    elif it <= 10000:
        optimizer.param_groups[0]['lr'] = 0.01
        optimizer.param_groups[1]['lr'] = 0.01
        optimizer.param_groups[2]['lr'] = 0.01
    elif it <= 20000:
        optimizer.param_groups[0]['lr'] = 0.001
        optimizer.param_groups[1]['lr'] = 0.001
        optimizer.param_groups[2]['lr'] = 0.001
    else:
        optimizer.param_groups[0]['lr'] = 0.0001
        optimizer.param_groups[1]['lr'] = 0.0001
        optimizer.param_groups[2]['lr'] = 0.0001
    
    # construct mini batch
    c = np.random.choice(range(len(train_class2ids)), replace=False, size=batch_size/2, p=class_probs)
    indices = []
    for i in c:
        indices = indices + np.random.choice(train_class2ids[i], replace=False, size=2).tolist()
        
    data, labels = construct_minibatch(train_imgs, train_labels, indices=indices, random_crop=True)
    data = torch.autograd.Variable(torch.from_numpy(data).float().cuda())
    labels = torch.autograd.Variable(torch.from_numpy(labels).long().cuda())
    
    # forward, backward & udpate
    net_feature.train(mode=True)
    optimizer.zero_grad()
    output = net_feature(data)
    loss = myloss(output, labels)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.data[0])
    
    # print
    if len(train_losses) > loss_frequency * 0 and (it % loss_frequency == 0):
        print 'Iter %d, elapsed time %f, loss %f' % (it, time.time() - tic, np.mean(train_losses[-loss_frequency:]))
        nams_logger.log('time_elapsed', time.time() - tic, step=it)
        nams_logger.log('train_loss_mean', np.mean(train_losses[-loss_frequency:]), step=it)
        nams_logger.flush()
        tic = time.time()
        
    if it % (test_frequency) == 0 and True:
        
        torch.save({
            'it': it,
            'model': net_feature,
            'loss': myloss,
            'optimizer': optimizer
        }, NAME + '.' + str(it) + '.pth')
        
        test_r1_full_all()
            












###############################################################
###############################################################
###############################################################















###############################################################
###############################################################
###############################################################



















