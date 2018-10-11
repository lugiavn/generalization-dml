import numpy as np
import scipy.misc
import torch
import math

def my_random_crop(img, size):
    assert(img.shape[0] >= size[0])
    assert(img.shape[1] >= size[1])
    start_index = np.random.randint(low=0, high=img.shape[0]-size[0]+1)
    img = img[start_index:start_index+size[0],:,:]
    start_index = np.random.randint(low=0, high=img.shape[1]-size[1]+1)
    img = img[:,start_index:start_index+size[1],:]
    return img

def my_resize_image(img, size, mode = 0):
    '''
    mode = 0: resize
    mode = 1: center crop, resize
    mode = 2: pad, resize
    '''
    if mode == 0:
        return scipy.misc.imresize(img, size)
    elif mode == 1:
        s = max(float(size[0]) / img.shape[0], float(size[1]) / img.shape[1])
        s = (img.shape[0] * s, img.shape[1] * s)
        s = (max(size[0], int(s[0])), max(size[1], int(s[1])))
        img = scipy.misc.imresize(img, s)
        assert img.shape[0] >= size[0], str(img.shape[0]) + ' and ' + str(size[0])
        assert(img.shape[1] >= size[1])
        if img.shape[0] > size[0]:
            start_index = int(img.shape[0] / 2.0 - size[0] / 2.0)
            img = img[start_index:start_index+size[0],:,:]
        if img.shape[1] > size[1]:
            start_index = int(img.shape[1] / 2.0 - size[1] / 2.0)
            img = img[:,start_index:start_index+size[1],:]
        return img
    else:
        assert(False)
        
        
def sind(x):
    return np.sin(np.deg2rad(x))
def cosd(x):
    return np.cos(np.deg2rad(x))

def gps_distance(lat1, long1, lat2, long2):
    delta_long = -(long1 - long2);
    delta_lat = -(lat1 - lat2);
    a = sind(delta_lat/2) * sind(delta_lat/2) + cosd(lat1) * cosd(lat2) * (sind(delta_long/2)*sind(delta_long/2));
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a));
    R = 6371;
    d = R * c;
    return d

class IM2GPSDataset(torch.utils.data.Dataset):
    def __init__(self, imagedatatxt):
        super(IM2GPSDataset, self).__init__()
        
        self.img_files = []
        self.labels = []
        with open(imagedatatxt, "r") as ins:
            array = []
            for line in ins:
                v = line.split()
                self.img_files.append(v[0])
                self.labels.append(v[1:])
                    
        self.labels = np.array(self.labels, dtype=np.float32)
        self.labelsGPS = self.labels[:,0:2]
        if self.labels.shape[1] >= 8:
            self.labels7k = np.array(self.labels[:,7], dtype=np.long)
        
        self.return_labelsGPS = False
        self.return_randomcrop = True
        self.return_centercrop = False
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index):
        
        # read image
        img = scipy.misc.imread(self.img_files[index])
        if len(img.shape) < 3:
            img1 = img
            img = np.zeros((img.shape[0], img.shape[1], 3))
            img[:,:,0] = img1
            img[:,:,1] = img1
            img[:,:,2] = img1
            
        # resize
        if self.return_centercrop:
            img = my_resize_image(img, (224, 224), mode=1)
        elif self.return_randomcrop:
            s = max(224.0/img.shape[0], 224.0/img.shape[1])
            s *= 1.0 + np.random.rand() * 0.1
            img = scipy.misc.imresize(img, s)
            img = my_random_crop(img, (224, 224))
        else:
            img = scipy.misc.imresize(img, (224, 224))
            
        # return
        img = (img.transpose([2, 0, 1]) / 255.0 - 0.45) / 0.224
        img = np.array(img, dtype=np.float32)
        if self.return_labelsGPS:
            label = self.labelsGPS[index,:]
        else:
            label = self.labels7k[index]
        return img, label


def test_retrieval_r1(features, labels, dot_product = True):
    # normalize features
    for i in range(features.shape[0]):
        features[i,:] = features[i,:] / np.linalg.norm(features[i,:])
        
    # check
    correct_count = 0.0
    sims = features.dot(features.T)
    if not dot_product:
        for i in range(features.shape[0]):
            for j in range(features.shape[0]):
                sims[i,j] = -np.linalg.norm(features[i,:]-features[j,:])
    for i in range(features.shape[0]):
        sims[i,i] = -99999
        j = np.argmax(sims[i,:])
        best_class = labels[j]
        if best_class == labels[i]:
            correct_count += 1
    return correct_count / features.shape[0]



##########################################################
##########################################################
##########################################################
import time
class NamLogger:
    def __init__(self, log_file_path):
        self.file_ = open(log_file_path, "w")
    
    def log(self, name, value, step = -1, flush = False):
        t = ''
        t += str(name)
        t += ';'
        t += str(type(value))
        t += ';'
        t += str(value)
        t += ';'
        t += str(step)
        t += ';'
        t += str(time.time())
        t += '\n'
        self.file_.write(t)
        if flush:
            self.flush()
    
    def flush(self):
        self.file_.flush()
        pass
    
    def close(self):
        self.file_.close()


##########################################################
##########################################################
##########################################################

import scipy.cluster
def NMI(a, b):
    def I(a,b):
        assert(len(a) == len(b))
        r = 0.0
        N = float(len(a))
        for i in np.unique(a):
            for j in np.unique(b):
                x = 0.0 + np.sum(a == i)
                y = 0.0 + np.sum(b == j)
                z = 0.0 + np.sum((a == i) * (b == j))
                if z > 0:
                    r += z / N * np.log(N * z / x / y)
        return r

    def H(a):
        r = 0.0
        N = float(len(a))
        for i in np.unique(a):
            x = 0.0 + np.sum(a == i)
            if x > 0:
                r += -x / N * np.log(x / N)
        return r
    
    return 2 * I(a, b) / (H(a) + H(b))


##########################################################
##########################################################
##########################################################

def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    N = labels.size(0)
    D = num_classes
    y = torch.zeros(N,D)
    j = 0
    for i in labels.data:
        y[j,i] = 1
        j += 1
    return y

class FocalLoss(torch.nn.Module):
    
    def __init__(self, gamma):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, data, target):
        
        data = torch.exp(data)
        p = data / torch.sum(data, dim=1, keepdim=True)
        x = one_hot_embedding(target, 110)
        x = torch.autograd.Variable(x.double().cuda())
        p = torch.sum(torch.mul(p, x), dim=1)
        loss = -torch.log(p)
        loss = torch.mul(loss, 1-p)
        loss = torch.mean(loss)
        return loss
        
##########################################################
##########################################################
##########################################################


import torch.autograd

class MyContrastiveLossFunc(torch.autograd.Function):

    def forward(self, features, labels):
        #print '=====my contrastive loss forward', features, type(features)
        m = 0.2
        self.save_for_backward(features, labels)
        features_np = features.cpu().numpy()
        labels_np = labels.cpu().numpy()
        pos_loss = 0.0
        neg_loss = 0.0
        pos_count = 0.0
        neg_count = 0.0
        distances = []
        for i in range(features_np.shape[0]):
            for j in range(features_np.shape[0]):
              if i != j:
                d = np.linalg.norm(features_np[i,:] - features_np[j,:])
                d = d * d
                distances.append(d)
                if labels_np[i] == labels_np[j]:
                    pos_count += 1
                    pos_loss += d
                elif labels_np[i] != labels_np[j] and d < m:
                    neg_count += 1
                    neg_loss += max(0, m - d)
        pos_loss /= pos_count + 1e-10
        neg_loss /= neg_count + 1e-10
        loss = (pos_loss + neg_loss) / 2.0
        #print 'avg distance', np.mean(distances), 'loss', loss, 'poss count', pos_count, 'neg count', neg_count
        loss = torch.FloatTensor((loss,))
        self.pos_count = pos_count
        self.neg_count = neg_count
        return loss
    
    def backward(self, grad_output):
        #print '------my contrastive loss backward', grad_output, type(grad_output)
        m = 0.2
        features, labels = self.saved_tensors
        features_np = features.cpu().numpy()
        labels_np = labels.cpu().numpy()
        grad_features = features.clone() * 0.0
        grad_features_np = grad_features.cpu().numpy()
        for i in range(features_np.shape[0]):
            for j in range(features_np.shape[0]):
              if i != j:
                d = np.linalg.norm(features_np[i,:] - features_np[j,:])
                d = d * d
                if labels_np[i] == labels_np[j]:
                        grad_features_np[i,:] += 2 * (features_np[i,:] - features_np[j,:]) / self.pos_count / 2.0
                        grad_features_np[j,:] += 2 * (features_np[j,:] - features_np[i,:]) / self.pos_count / 2.0
                elif labels_np[i] != labels_np[j] and d < m:
                        grad_features_np[i,:] += -2 * (features_np[i,:] - features_np[j,:]) / self.neg_count / 2.0
                        grad_features_np[j,:] += -2 * (features_np[j,:] - features_np[i,:]) / self.neg_count / 2.0
        
        for i in range(features_np.shape[0]):
            for k in range(features_np.shape[1]):
                grad_features[i,k] = float(grad_features_np[i,k])
        return grad_features, None

import scipy.stats


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    source: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)

class MyTripletLossFunc(torch.autograd.Function):
    
    def __init__(self, DBL = True, m = 0.2, triplet_type = 0):
        super(MyTripletLossFunc, self).__init__()
        self.DBL = DBL
        self.m = m
        self.distance_weighting = False
        self.triplet_type = triplet_type
        
    def forward(self, features, labels):
        self.save_for_backward(features, labels)
        features_np = features.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        # compute distances
        self.distances = np.zeros((features_np.shape[0],features_np.shape[0]))
        if True:
            self.distances = pairwise_distances(features).cpu().numpy()
        else:
            for i in range(features_np.shape[0]):
                for j in range(features_np.shape[0]):
                    d = np.linalg.norm(features_np[i,:] - features_np[j,:])
                    d = d * d
                    self.distances[i,j] = d
                
        distance_mean = np.mean(self.distances)
        distance_std = np.std(self.distances)
                
        loss = 0.0
        triplet_count = 0.0
        correct_count = 0.0
        for i in range(features_np.shape[0]):
         for j in range(features_np.shape[0]):
          for k in range(features_np.shape[0]):
            cond = i != j and labels_np[i] == labels_np[j] and labels_np[i] != labels_np[k]
            if self.triplet_type == 1:
                cond = i != j and labels_np[i] == labels_np[j] and self.distances[i,j] < self.distances[i,k]
            if self.triplet_type == 2:
                cond = labels_np[i] != labels_np[j] and labels_np[i] != labels_np[k] and self.distances[i,j] > self.distances[i,k]
            if cond:
                w = 1.0
                if self.distance_weighting:
                    w = 1.0 / scipy.stats.norm.pdf(self.distances[i,k], distance_mean, distance_std)
                triplet_count += w
                if self.DBL:
                    loss += w * np.log(1 + np.exp(self.distances[i,j] - self.distances[i,k]))
                else:
                    loss += w * max(0, self.distances[i,j] - self.distances[i,k] + self.m)
                if self.distances[i,j] < self.distances[i,k]:
                    correct_count += 1
        loss /= triplet_count
        self.triplet_count = triplet_count
        self.debug_str = 'Loss ' + str(loss)
        self.debug_str += ', triplet count ' + str(triplet_count) 
        self.debug_str += ', accuracy ' + str(correct_count/triplet_count)
        self.debug_str += ', avg distance ' + str(np.mean(self.distances))
        return torch.FloatTensor((loss,))
    
    def backward(self, grad_output):
        features, labels = self.saved_tensors
        features_np = features.cpu().numpy()
        labels_np = labels.cpu().numpy()
        grad_features = features.clone() * 0.0
        grad_features_np = grad_features.cpu().numpy()
        
        distance_mean = np.mean(self.distances)
        distance_std = np.std(self.distances)
        
        for i in range(features_np.shape[0]):
         for j in range(features_np.shape[0]):
          for k in range(features_np.shape[0]):
            cond = i != j and labels_np[i] == labels_np[j] and labels_np[i] != labels_np[k]
            if self.triplet_type == 1:
                cond = i != j and labels_np[i] == labels_np[j] and self.distances[i,j] < self.distances[i,k]
            if self.triplet_type == 2:
                cond = labels_np[i] != labels_np[j] and labels_np[i] != labels_np[k] and self.distances[i,j] > self.distances[i,k]
            if cond:
                w = 1.0
                if self.distance_weighting:
                    w = 1.0 / scipy.stats.norm.pdf(self.distances[i,k], distance_mean, distance_std)
                
                f = 0.0
                if self.distances[i,j] - self.distances[i,k] + self.m > 0:
                    f = 1.0
                if self.DBL:
                    f = 1.0 - 1.0 / (1.0 + np.exp(self.distances[i,j] - self.distances[i,k]))
                grad_features_np[i,:] += w * f * (features_np[i,:] - features_np[j,:]) / self.triplet_count
                grad_features_np[j,:] += w * f * (features_np[j,:] - features_np[i,:]) / self.triplet_count
                grad_features_np[i,:] += -w * f * (features_np[i,:] - features_np[k,:]) / self.triplet_count
                grad_features_np[k,:] += -w * f * (features_np[k,:] - features_np[i,:]) / self.triplet_count
                    
        for i in range(features_np.shape[0]):
            #for k in range(features_np.shape[1]):
            #   grad_features[i,k] = float(grad_features_np[i,k])
            grad_features[i,:] = torch.from_numpy(grad_features_np[i,:])
        return grad_features, None
    
class MyTripletLoss(torch.nn.Module):
    def __init__(self, normalize_scale = 3.0, learn_scale = False, DBL = True, m = 0.2, triplet_type = 0):
        super(MyTripletLoss, self).__init__()
        self.DBL = DBL
        self.m = m
        self.norm_s = float(normalize_scale)
        if learn_scale:
            self.norm_s = torch.nn.Parameter(torch.FloatTensor((self.norm_s,)))
        self.triplet_type = triplet_type
        
    def forward(self, x, labels):
        features = self.norm_s * x / torch.norm(x, dim=1, keepdim=True).expand_as(x)
        loss = MyTripletLossFunc(DBL=self.DBL, m=self.m, triplet_type=self.triplet_type)(features, labels)
        return loss


class MyContrastiveLoss(torch.nn.Module):
    def __init__(self, normalize_scale = 3.0, learn_scale = False):
        super(MyContrastiveLoss, self).__init__()
        self.norm_s = float(normalize_scale)
        if learn_scale:
            self.norm_s = torch.nn.Parameter(torch.FloatTensor((self.norm_s,)))
        
    def forward(self, x, labels):
        features = self.norm_s * x / torch.norm(x, dim=1, keepdim=True).expand_as(x)
        loss = MyContrastiveLossFunc()(features, labels)
        return loss




class MyDistancesFunc(torch.autograd.Function):
    
    def __init__(self):
        super(MyDistancesFunc, self).__init__()
        
    def forward(self, features1, features2):
        self.save_for_backward(features1, features2)
        features1_np = features1.cpu().numpy()
        features2_np = features2.cpu().numpy()
        self.distances = np.zeros((features1_np.shape[0], features2_np.shape[0]))
        for i in range(features1_np.shape[0]):
            for j in range(features2_np.shape[0]):
                d = np.linalg.norm(features1_np[i,:] - features2_np[j,:])
                self.distances[i,j] = d * d
        return torch.from_numpy(self.distances).cuda().clone()
    
    def backward(self, grad_output):
        features1, features2 = self.saved_tensors
        features1_np = features1.cpu().numpy()
        features2_np = features2.cpu().numpy()
        grad_features1 = features1.clone() * 0.0
        grad_features2 = features2.clone() * 0.0
        grad_features1_np = grad_features1.cpu().numpy()
        grad_features2_np = grad_features2.cpu().numpy()
        grad_distances_np = grad_output.cpu().numpy()
        for i in range(features1_np.shape[0]):
            for j in range(features2_np.shape[0]):
                grad_features1_np[i,:] += grad_distances_np[i,j] * (features1_np[i,:] - features2_np[j,:])
                grad_features2_np[j,:] += grad_distances_np[i,j] * (features2_np[j,:] - features1_np[i,:])
               
        return torch.from_numpy(grad_features1_np).cuda().clone(), torch.from_numpy(grad_features2_np).cuda().clone()

class MyProxyNCA(torch.nn.Module):
    def __init__(self, num_proxies = 110, feature_size = 128, normalize_scale = 3.0, learn_scale = True):
        super(MyProxyNCA, self).__init__()
        self.proxies = torch.nn.Parameter(torch.randn(num_proxies, feature_size) / 8)
        self.norm_s = float(normalize_scale)
        if learn_scale:
            self.norm_s = torch.nn.Parameter(torch.FloatTensor((self.norm_s,)))
        
    def forward(self, x, labels):
        proxies = self.proxies
        #proxies = self.norm_s * self.proxies / torch.norm(self.proxies, dim=1, keepdim=True).expand_as(self.proxies)
        features = self.norm_s * x / torch.norm(x, dim=1, keepdim=True).expand_as(x)
        #distances = MyDistancesFunc()(features, proxies)
        #sims = -distances
        sims = torch.mm(features, torch.t(proxies))
        loss = torch.nn.CrossEntropyLoss()(sims, labels)
        #loss = FocalLoss(gamma=1.0)(sims, labels)
        return loss


class MyFeatureClassification(torch.nn.Module):
    def __init__(self, in_features, num_classes):
        super(MyFeatureClassification, self).__init__()
        self.num_classes = num_classes
        self.fc = torch.nn.Linear(in_features, num_classes, bias=False)
        
    def forward(self, x, labels):
        x = x / torch.norm(x, dim=1, keepdim=True).expand_as(x)
        x = self.fc(x)
        loss = torch.nn.CrossEntropyLoss()(x, labels)
        return loss
    




















