import urllib
import os
import logging

import numpy as np
import PIL.Image
from google.protobuf import text_format

import caffe

def ensure_file(path, url):
    if not os.path.isfile(path):
        print 'Local file %s not found, downloading from %s' % (path, url)
        urllib.urlretrieve(url, path)

net_fn = 'deploy.prototxt'
ensure_file(net_fn, 'https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/'
    'bb2b4fe0a9bb0669211cf3d0bc949dfdda173e9e/VGG_ILSVRC_19_layers_deploy.prototxt')

param_fn = 'vggnet.caffemodel'
ensure_file(param_fn, 'http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel')

imagenet_mean = np.float32([104.0, 116.0, 122.0])

#original model requires some patching
model = caffe.io.caffe_pb2.NetParameter()
text_format.Merge(open(net_fn).read(), model)
model.force_backward = True

#FIXME: do not use hardcoded constants
for layer in model.layers:
    if layer.type == 17:
        print layer.pooling_param.pool
        layer.pooling_param.pool = 1

with open('tmp.prototxt', 'w') as f:
    f.write(str(model))

net = caffe.Classifier('tmp.prototxt', param_fn,
                       mean=imagenet_mean,
                       channel_swap=(2,1,0))

def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']

def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])

def gram_matrix(activations):
    channels = activations.shape[0]
    gram = np.zeros((channels, channels), dtype=np.float32)
    for i in xrange(channels):
        for j in xrange(channels):
            gram[i, j] = np.mean(activations[i] * activations[j])
    return gram / (channels ** 2)

def style_representation(img, net, layers):
    src = net.blobs['data']
    src.reshape(1, 3, img.shape[1], img.shape[2])
    src.data[0] = img
    net.forward(end=layers[-1])

    activations = []
    for layer in layers:
        dst = net.blobs[layer]
        activations.append(dst.data[0])

    return [gram_matrix(activation) for activation in activations]

def style_grad(style, activations):
    gram = gram_matrix(activations)
    d = style - gram
    loss = np.mean(d * d)
    print "Style loss %.5g" % loss
    shape = activations.shape
    activations = activations.reshape(shape[0], shape[1] * shape[2])

    result = (np.dot(d, activations)).reshape(shape[0], shape[1], shape[2]) #/ (shape[1] * shape[2])
    return result

def info_representation(img, net, end):
    src = net.blobs['data']
    dst = net.blobs[end]

    src.reshape(1, 3, img.shape[1], img.shape[2])
    src.data[0] = img
    net.forward(end=end)
    return np.copy(dst.data[0])

def info_grad(info, activations):
    d = info - activations
    loss = d * d
    print "Info loss %.5g" % np.mean(loss)

    return d

class Adadelta(object):
    def __init__(self):
        self.decay = 0.05
        self.epsilon = 1e-6
        
    def get_update(self, gradient):
        if not hasattr(self, 'accumulated_gradient'):
            self.accumulated_gradient = np.zeros_like(gradient)
            self.accumulated_update = np.zeros_like(gradient)

        self.accumulated_gradient = (1 - self.decay) * self.accumulated_gradient + self.decay * gradient * gradient
        update = np.power(self.accumulated_update + self.epsilon, 0.5) / np.power(self.accumulated_update + self.epsilon, 0.5) * gradient
        self.accumulated_update = (1 - self.decay) * self.accumulated_update + self.decay * update * update

        return update

layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1']#, 'conv5_1']
prev_layers = ['data'] + layers[:-1]
weights = [1.0 / len(layers)] * len(layers)

info_layer = 'conv3_1'
info_weight = 10.

optimizer = Adadelta()

style_img = np.float32(PIL.Image.open('cat.jpg'))
styles = style_representation(preprocess(net, style_img), net, layers)

info_img = np.float32(PIL.Image.open('hawk.jpg'))
info = info_representation(preprocess(net, info_img), net, info_layer)

means = [np.mean(channel) for channel in preprocess(net, style_img)]
img = np.asarray(means, dtype=np.float32) + np.random.uniform(low=-30., high=30., size=info_img.shape) + imagenet_mean
src = net.blobs['data']
data = preprocess(net, img)
src.reshape(1, *data.shape)
src.data[0] = data

logging.basicConfig(level=logging.INFO)

for i in xrange(400):
    net.forward(end=layers[-1])
    net.blobs[layers[-1]].diff[0] = 0

    for layer, prev, style, weight in reversed(zip(layers, prev_layers, styles, weights)):
        dst = net.blobs[layer]
        dst.diff[0] += weight * style_grad(style, dst.data[0])
        if layer == info_layer: 
            dst.diff[0] += info_weight * info_grad(info, dst.data[0])

        if prev != 'data':
            net.backward(start=layer, end=prev)
        else:
            net.backward(start=layer)
    
    src = net.blobs['data']
    grad = src.diff[0]
    update = optimizer.get_update(1e-6 * grad)
    src.data[0] += update
    logging.info("Epoch %d completed, average update %.2g" % (i, np.mean(update)))
    bias = net.transformer.mean['data']
    src.data[:] = np.clip(src.data[0], -bias, 255-bias)
    PIL.Image.fromarray(np.uint8(deprocess(net, src.data[0]))).save("%d.jpg" % i)
