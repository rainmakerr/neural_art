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

def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']

def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])

def gram_matrix(activations):
    shape = activations.shape
    reshaped = activations.reshape(shape[0], shape[1] * shape[2])
    gram = np.dot(reshaped, reshaped.T)

    return gram / (shape[0] * shape[1] * shape[2]) ** 2

def style_representation(net, img, layers):
    img = preprocess(net, img)
    src = net.blobs['data']
    src.reshape(1, *img.shape)
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
    loss = np.sum(d * d)
    logging.info('Style loss %.5g', loss)
    shape = activations.shape
    activations = activations.reshape(shape[0], shape[1] * shape[2])

    result = (np.dot(d, activations)).reshape(shape[0], shape[1], shape[2])
    return result

def info_representation(net, img, end):
    img = preprocess(net, img)
    src = net.blobs['data']
    dst = net.blobs[end]

    src.reshape(1, *img.shape)
    src.data[0] = img
    net.forward(end=end)
    return np.copy(dst.data[0])

def info_grad(info, activations):
    d = info - activations
    loss = d * d
    logging.info('Info loss %.5g', np.sum(loss))

    return d

class Adadelta(object):
    def __init__(self, decay=0.05, epsilon=1e-6):
        self.decay = decay
        self.epsilon = epsilon
        
    def get_update(self, gradient):
        if not hasattr(self, 'accumulated_gradient'):
            self.accumulated_gradient = np.zeros_like(gradient)
            self.accumulated_update = np.zeros_like(gradient)

        self.accumulated_gradient = (1 - self.decay) * self.accumulated_gradient + self.decay * gradient * gradient
        update = np.sqrt(self.accumulated_update + self.epsilon) / np.sqrt(self.accumulated_gradient + self.epsilon) * gradient
        self.accumulated_update = (1 - self.decay) * self.accumulated_update + self.decay * update * update

        logging.info('Accumulated update: %.5g, gradient: %.5g', np.mean(self.accumulated_update), np.mean(self.accumulated_gradient))

        return 1000 * update

def combine_images(info_image, style_image, models_dir, output_dir, info_weight=1e-7, iterations=400):
    np.random.seed(1337)

    layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
    prev_layers = ['data'] + layers[:-1]
    unnormalized_weights = [1.0, 1.5, 3.0, 5.0, 50.0]
    weights = [w / len(layers) for w in unnormalized_weights]

    info_layer = 'relu4_1'

    optimizer = Adadelta()
    imagenet_mean = np.float32([104.0, 116.0, 122.0])

    net_fn = os.path.join(models_dir, 'deploy.prototxt')
    ensure_file(net_fn, 'https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/'
        'bb2b4fe0a9bb0669211cf3d0bc949dfdda173e9e/VGG_ILSVRC_19_layers_deploy.prototxt')

    param_fn = os.path.join(models_dir, 'vggnet.caffemodel')
    ensure_file(param_fn, 'http://bethgelab.org/media/uploads/deeptextures/vgg_normalised.caffemodel')

    #original model requires some patching
    #first, allow gradient propagation to data blob
    model = caffe.io.caffe_pb2.NetParameter()
    text_format.Merge(open(net_fn).read(), model)
    model.force_backward = True

    blob_substitutions = {}
    for layer in model.layers:
        #paper suggests using average pooling instead of max
        if layer.type == caffe.io.caffe_pb2.V1LayerParameter.POOLING:
            layer.pooling_param.pool = caffe.io.caffe_pb2.PoolingParameter.AVE
        #separating convolution and relu blobs removes problems with partial forward passes at cost of some memory
        if layer.bottom[0] in blob_substitutions.keys():
            layer.bottom[0] = blob_substitutions[layer.bottom[0]]
        if layer.type == caffe.io.caffe_pb2.V1LayerParameter.RELU and not layer.bottom[0].startswith('fc'):
            new_top_blob = layer.top[0].replace('conv', 'relu')
            blob_substitutions[layer.top[0]] = new_top_blob
            layer.top[0] = new_top_blob

    tmp_proto = os.path.join(models_dir, 'tmp.prototxt')
    with open(tmp_proto, 'w') as f:
        f.write(str(model))

    net = caffe.Classifier(tmp_proto, param_fn,
                           mean=imagenet_mean,
                           channel_swap=(2,1,0))

    styles = style_representation(net, style_image, layers)

    info = info_representation(net, info_image, info_layer)

    img = np.random.uniform(low=0., high=255., size=info_image.shape)
    src = net.blobs['data']
    data = preprocess(net, img)
    src.reshape(1, *data.shape)
    src.data[0] = data

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    for i in range(iterations):
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
        update = optimizer.get_update(1e7 * grad)
        src.data[0] += update
        logging.info('Iteration %d completed', i)
        bias = net.transformer.mean['data']
        src.data[0] = np.clip(src.data[0], -bias, 255-bias)

        if output_dir is not None:
            filename = os.path.join(output_dir, '%d.jpg' % i)
            PIL.Image.fromarray(np.uint8(deprocess(net, src.data[0]))).save(filename)

    return deprocess(net, src.data[0])