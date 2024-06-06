# Copyright 2018 The Lucid Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import warnings

import numpy as np
import tensorflow as tf
from cachetools.func import lru_cache

from lucid_kietzmannlab.misc.io import loading
from lucid_kietzmannlab.modelzoo.vision_base import Model

PATH_TEMPLATE = "gs://modelzoo/aligned-activations/{}/{}-{:05d}-of-01000.npy"
PAGE_SIZE = 10000
NUMBER_OF_AVAILABLE_SAMPLES = 100000
assert NUMBER_OF_AVAILABLE_SAMPLES % PAGE_SIZE == 0
NUMBER_OF_PAGES = NUMBER_OF_AVAILABLE_SAMPLES // PAGE_SIZE
IMAGENET_MEAN = np.array([123.68, 116.779, 103.939])
IMAGENET_MEAN_BGR = np.flip(IMAGENET_MEAN, 0)


class Layer:
    """Layer provides information on a model's layers."""

    width = None  # reserved for future use
    height = None  # reserved for future use
    shape = None  # reserved for future use

    def __init__(self, model_instance: "Model", name, depth, tags):
        self._activations = None
        self.model_class = model_instance.__class__
        self.model_name = model_instance.name
        self.name = name
        self.depth = depth
        self.tags = set(tags)

    def __getitem__(self, name):
        if name == "type":
            warnings.warn(
                "Property 'type' is deprecated on model layers. Please check if 'tags' contains the type you are looking for in the future! We're simply a tag for now.",
                DeprecationWarning,
            )
            return list(self.tags)[0]
        if name not in self.__dict__:
            error_message = f"'Layer' object has no attribute '{name}'"
            raise AttributeError(error_message)
        return self.__dict__[name]

    @property
    def size(self):
        warnings.warn(
            "Property 'size' is deprecated on model layers because it may be confused with the spatial 'size' of a layer. Please use 'depth' in the future!",
            DeprecationWarning,
        )
        return self.depth

    @property
    def activations(self):
        """Loads sampled activations, which requires network access."""
        if self._activations is None:
            self._activations = _get_aligned_activations(self)
        return self._activations

    def __repr__(self):
        return f"Layer (belonging to {self.model_name}) <{self.name}: {self.depth}> ([{self.tags}])"

    def to_json(self):
        return self.name  # TODO


@lru_cache()
def _get_aligned_activations(layer) -> np.ndarray:
    """Downloads 100k activations of the specified layer sampled from iterating over
    ImageNet. Activations of all layers where sampled at the same spatial positions for
    each image, allowing the calculation of correlations."""
    activation_paths = [
        PATH_TEMPLATE.format(
            sanitize(layer.model_name), sanitize(layer.name), page
        )
        for page in range(NUMBER_OF_PAGES)
    ]
    activations = np.vstack([loading.load(path) for path in activation_paths])
    assert np.all(np.isfinite(activations))
    return activations


@lru_cache()
def layer_covariance(layer1, layer2=None):
    """Computes the covariance matrix between the neurons of two layers. If only one
    layer is passed, computes the symmetric covariance matrix of that layer."""
    layer2 = layer2 or layer1
    act1, act2 = layer1.activations, layer2.activations
    num_datapoints = act1.shape[
        0
    ]  # cast to avoid numpy type promotion during division
    return np.matmul(act1.T, act2) / float(num_datapoints)


@lru_cache()
def layer_inverse_covariance(layer):
    """Inverse of a layer's correlation matrix. Function exists mostly for caching."""
    return np.linalg.inv(layer_covariance(layer))


def push_activations(activations, from_layer, to_layer):
    """Push activations from one model to another using prerecorded correlations"""
    inverse_covariance_matrix = layer_inverse_covariance(from_layer)
    activations_decorrelated = np.dot(
        inverse_covariance_matrix, activations.T
    ).T
    covariance_matrix = layer_covariance(from_layer, to_layer)
    activation_recorrelated = np.dot(
        activations_decorrelated, covariance_matrix
    )
    return activation_recorrelated


def _layers_from_list_of_dicts(model_instance: "Model", list_of_dicts):
    layers = []
    for layer_info in list_of_dicts:
        name, depth, tags = (
            layer_info["name"],
            layer_info["depth"],
            layer_info["tags"],
        )
        layer = Layer(model_instance, name, depth, tags)
        layers.append(layer)
    return tuple(layers)


def sanitize(string):
    return string.replace("/", "_")


@lru_cache()
def get_aligned_activations(layer) -> np.ndarray:
    """Downloads 100k activations of the specified layer sampled from iterating over
    ImageNet. Activations of all layers where sampled at the same spatial positions for
    each image, allowing the calculation of correlations."""
    activation_paths = [
        PATH_TEMPLATE.format(
            sanitize(layer.model_name), sanitize(layer.name), page
        )
        for page in range(NUMBER_OF_PAGES)
    ]
    activations = np.vstack([loading.load(path) for path in activation_paths])
    assert np.all(np.isfinite(activations))
    return activations


def populate_inception_bottlenecks(scope):
    """Add Inception bottlenecks and their pre-Relu versions to the graph."""
    graph = tf.compat.v1.get_default_graph()
    for op in graph.get_operations():
        if op.name.startswith(scope + "/") and "Concat" in op.type:
            name = op.name.split("/")[1]
            pre_relus = []
            for tower in op.inputs[1:]:
                if tower.op.type == "Relu":
                    tower = tower.op.inputs[0]
                pre_relus.append(tower)
            concat_name = scope + "/" + name + "_pre_relu"
            _ = tf.concat(pre_relus, -1, name=concat_name)


def _populate_inception_bottlenecks(scope):
    """Add Inception bottlenecks and their pre-Relu versions to the graph."""
    graph = tf.get_default_graph()
    for op in graph.get_operations():
        if op.name.startswith(scope + "/") and "Concat" in op.type:
            name = op.name.split("/")[1]
            pre_relus = []
            for tower in op.inputs[1:]:
                if tower.op.type == "Relu":
                    tower = tower.op.inputs[0]
                pre_relus.append(tower)
            concat_name = scope + "/" + name + "_pre_relu"
            _ = tf.concat(pre_relus, -1, name=concat_name)


class AlexNet(Model):
    """Original AlexNet weights ported to TF.

    AlexNet is the breakthrough vision model from Krizhevsky, et al (2012):
    https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
    This implementation is a caffe re-implementation:
    http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
    It was converted to TensorFlow by this GitHub project:
    https://github.com/huanzhang12/tensorflow-alexnet-model
    It appears the parameters are the actual original parameters.
    """

    # The authors of code to convert AlexNet to TF host weights at
    # http://jaina.cs.ucdavis.edu/datasets/adv/imagenet/alexnet_frozen.pb
    # but it seems more polite and reliable to host our own.
    model_path = "gs://modelzoo/vision/other_models/AlexNet.pb"
    labels_path = "gs://modelzoo/labels/ImageNet_standard.txt"
    synsets_path = "gs://modelzoo/labels/ImageNet_standard_synsets.txt"
    dataset = "ImageNet"
    image_shape = [227, 227, 3]
    is_BGR = True
    image_value_range = (-IMAGENET_MEAN_BGR, 255 - IMAGENET_MEAN_BGR)
    input_name = "Placeholder"

    def __init__(self, model_checkpoint_dir, model_checkpoint):

        self.model_checkpoint_dir = model_checkpoint_dir
        self.model_checkpoint = model_checkpoint
        self._graph_def = None

    def load_model_checkpoint(self, sess):
        saver = tf.compat.v1.train.import_meta_graph(
            f"{self.model_checkpoint_dir}/{self.model_checkpoint}.meta"
        )
        saver.restore(
            sess, f"{self.model_checkpoint_dir}/{self.model_checkpoint}"
        )

    @property
    def graph_def(self):
        if not self._graph_def:
            # Load the model checkpoint and return the graph_def

            with tf.compat.v1.Graph().as_default() as graph:
                with tf.compat.v1.Session() as sess:
                    self.load_model_checkpoint(sess)
                    self._graph_def = graph.as_graph_def()

        return self._graph_def


AlexNet.layers = _layers_from_list_of_dicts(
    AlexNet(None, None),
    [
        {"tags": ["pre_relu", "conv"], "name": "Conv2D", "depth": 96},
        {"tags": ["pre_relu", "conv"], "name": "Conv2D_1", "depth": 128},
        {"tags": ["pre_relu", "conv"], "name": "Conv2D_2", "depth": 128},
        {"tags": ["pre_relu", "conv"], "name": "Conv2D_3", "depth": 384},
        {"tags": ["pre_relu", "conv"], "name": "Conv2D_4", "depth": 192},
        {"tags": ["pre_relu", "conv"], "name": "Conv2D_5", "depth": 192},
        {"tags": ["pre_relu", "conv"], "name": "Conv2D_6", "depth": 128},
        {"tags": ["pre_relu", "conv"], "name": "Conv2D_7", "depth": 128},
        {"tags": ["dense"], "name": "Relu", "depth": 4096},
        {"tags": ["dense"], "name": "Relu_1", "depth": 4096},
        {"tags": ["dense"], "name": "Softmax", "depth": 1000},
    ],
)