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


import os
import re
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


def get_layer_names_tensors(model):

    if isinstance(model, AlexNet):

        with tf.Graph().as_default() as graph:

            with tf.compat.v1.Session(graph=graph) as sess:
                model.graph = sess.graph
                tf.import_graph_def(model.graph_def, name="")
                layer_shape_dict = {}
                input_operations = [
                    op.name for op in model.graph.get_operations()
                ]
                for tensor_name in input_operations:
                    tensor_shape = model.graph.get_tensor_by_name(
                        f"{tensor_name}:0"
                    )
                    layer_shape_dict[tensor_name] = tensor_shape

    if isinstance(model, EcoAlexModel):

        with tf.Graph().as_default() as graph:

            with tf.compat.v1.Session(graph=graph) as sess:
                model.graph = sess.graph
                tf.import_graph_def(model.graph_def, name="")
                layer_shape_dict = {}
                # Load the model
                checkpoint_path = os.path.join(
                    model.model_checkpoint_dir, model.model_checkpoint
                )
                meta_path = os.path.join(
                    model.model_checkpoint_dir,
                    f"{model.model_checkpoint}.meta",
                )

                saver = tf.compat.v1.train.import_meta_graph(
                    meta_path, clear_devices=True
                )
                saver.restore(sess, checkpoint_path)
                input_operations = [op for op in model.graph.get_operations()]
                input_tensor_names = []
                for op in input_operations:
                    input_tensor_names.extend(
                        [output_tensor.name for output_tensor in op.outputs]
                    )

                pattern = r"(tower_\d+/.*?/(?:conv|fc)\d*/.*?Conv2D)"

                # Set to store the matching layer names
                matching_layer_names = set()

                # Traverse the graph to find operations matching the pattern
                for op in model.graph.get_operations():
                    match = re.search(pattern, op.name)
                    if match:
                        matching_layer_names.add(match.group(1))
                matching_layer_names = {
                    layer_name
                    for layer_name in matching_layer_names
                    if "gradients" not in layer_name
                }

                for tensor_name in sorted(matching_layer_names):
                    tensor_shape = model.graph.get_tensor_by_name(
                        f"{tensor_name}:0"
                    )
                    layer_shape_dict[tensor_name] = tensor_shape
                print("Model loaded from:", model.model_checkpoint_dir)

    return layer_shape_dict


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
    image_shape = [112, 112, 3]
    is_BGR = True
    image_value_range = (-IMAGENET_MEAN_BGR, 255 - IMAGENET_MEAN_BGR)
    input_name = "Placeholder"


AlexNet.layers = _layers_from_list_of_dicts(
    AlexNet(),
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


class EcoAlexModel(Model):

    def __init__(self, model_checkpoint_dir, model_checkpoint):
        self.image_shape = [3, 112, 112]
        self.dataset = "ImageNet"

        self.is_BGR = False
        self.image_value_range = (-IMAGENET_MEAN_BGR, 255 - IMAGENET_MEAN_BGR)
        self.input_name = "Placeholder"
        self.model_checkpoint_dir = model_checkpoint_dir
        self.model_checkpoint = model_checkpoint

    @property
    def graph_def(self):
        """Returns the serialized GraphDef representation of the TensorFlow graph."""

        return self.graph.as_graph_def()

    def load_model_layers(self):
        self.layers = _layers_from_list_of_dicts(
            self,
            [
                {
                    "tags": ["conv"],
                    "name": "tower_0/alexnet_v2/conv1/Conv2D",
                    "depth": 64,
                },
                {
                    "tags": ["conv"],
                    "name": "tower_0/alexnet_v2/conv2/Conv2D",
                    "depth": 192,
                },
                {
                    "tags": ["conv"],
                    "name": "tower_0/alexnet_v2/conv3/Conv2D",
                    "depth": 384,
                },
                {
                    "tags": ["conv"],
                    "name": "tower_0/alexnet_v2/conv4/Conv2D",
                    "depth": 384,
                },
                {
                    "tags": ["conv"],
                    "name": "tower_0/alexnet_v2/conv5/Conv2D",
                    "depth": 256,
                },
                {
                    "tags": ["dense"],
                    "name": "tower_0/alexnet_v2/fc6/Conv2D",
                    "depth": 4096,
                },
                {
                    "tags": ["dense"],
                    "name": "tower_0/alexnet_v2/fc7/Conv2D",
                    "depth": 4096,
                },
                {
                    "tags": ["dense"],
                    "name": "tower_0/alexnet_v2/fc8/Conv2D",
                    "depth": 565,
                },
                {
                    "tags": ["conv"],
                    "name": "tower_1/alexnet_v2/conv1/Conv2D",
                    "depth": 64,
                },
                {
                    "tags": ["conv"],
                    "name": "tower_1/alexnet_v2/conv2/Conv2D",
                    "depth": 192,
                },
                {
                    "tags": ["conv"],
                    "name": "tower_1/alexnet_v2/conv3/Conv2D",
                    "depth": 384,
                },
                {
                    "tags": ["conv"],
                    "name": "tower_1/alexnet_v2/conv4/Conv2D",
                    "depth": 384,
                },
                {
                    "tags": ["conv"],
                    "name": "tower_1/alexnet_v2/conv5/Conv2D",
                    "depth": 256,
                },
                {
                    "tags": ["dense"],
                    "name": "tower_1/alexnet_v2/fc6/Conv2D",
                    "depth": 4096,
                },
                {
                    "tags": ["dense"],
                    "name": "tower_1/alexnet_v2/fc7/Conv2D",
                    "depth": 4096,
                },
                {
                    "tags": ["dense"],
                    "name": "tower_1/alexnet_v2/fc8/Conv2D",
                    "depth": 565,
                },
                {
                    "tags": ["conv"],
                    "name": "tower_2/alexnet_v2/conv1/Conv2D",
                    "depth": 64,
                },
                {
                    "tags": ["conv"],
                    "name": "tower_2/alexnet_v2/conv2/Conv2D",
                    "depth": 192,
                },
                {
                    "tags": ["conv"],
                    "name": "tower_2/alexnet_v2/conv3/Conv2D",
                    "depth": 384,
                },
                {
                    "tags": ["conv"],
                    "name": "tower_2/alexnet_v2/conv4/Conv2D",
                    "depth": 384,
                },
                {
                    "tags": ["conv"],
                    "name": "tower_2/alexnet_v2/conv5/Conv2D",
                    "depth": 256,
                },
                {
                    "tags": ["dense"],
                    "name": "tower_2/alexnet_v2/fc6/Conv2D",
                    "depth": 4096,
                },
                {
                    "tags": ["dense"],
                    "name": "tower_2/alexnet_v2/fc7/Conv2D",
                    "depth": 4096,
                },
                {
                    "tags": ["dense"],
                    "name": "tower_2/alexnet_v2/fc8/Conv2D",
                    "depth": 565,
                },
                {
                    "tags": ["conv"],
                    "name": "tower_3/alexnet_v2/conv1/Conv2D",
                    "depth": 64,
                },
                {
                    "tags": ["conv"],
                    "name": "tower_3/alexnet_v2/conv2/Conv2D",
                    "depth": 192,
                },
                {
                    "tags": ["conv"],
                    "name": "tower_3/alexnet_v2/conv3/Conv2D",
                    "depth": 384,
                },
                {
                    "tags": ["conv"],
                    "name": "tower_3/alexnet_v2/conv4/Conv2D",
                    "depth": 384,
                },
                {
                    "tags": ["conv"],
                    "name": "tower_3/alexnet_v2/conv5/Conv2D",
                    "depth": 256,
                },
                {
                    "tags": ["dense"],
                    "name": "tower_3/alexnet_v2/fc6/Conv2D",
                    "depth": 4096,
                },
                {
                    "tags": ["dense"],
                    "name": "tower_3/alexnet_v2/fc7/Conv2D",
                    "depth": 4096,
                },
                {
                    "tags": ["dense"],
                    "name": "tower_3/alexnet_v2/fc8/Conv2D",
                    "depth": 565,
                },
            ],
        )

    def load_ecoset_model_seeds(self):

        with tf.Graph().as_default() as graph:

            with tf.compat.v1.Session(graph=graph) as sess:
                self.graph = sess.graph
                tf.import_graph_def(self.graph_def, name="")
                layer_shape_dict = {}
                # Load the model
                checkpoint_path = os.path.join(
                    self.model_checkpoint_dir, self.model_checkpoint
                )
                meta_path = os.path.join(
                    self.model_checkpoint_dir, f"{self.model_checkpoint}.meta"
                )

                saver = tf.compat.v1.train.import_meta_graph(
                    meta_path, clear_devices=True
                )
                saver.restore(sess, checkpoint_path)
                print("Model loaded from:", self.model_checkpoint_dir)

                # Find operations with no incoming edges (i.e., the inputs to the graph)
                input_operations = [op for op in self.graph.get_operations()]

                # Get the output tensor names of the input operations
                input_tensor_names = []
                for op in input_operations:
                    input_tensor_names.extend(
                        [output_tensor.name for output_tensor in op.outputs]
                    )

                pattern = r"(tower_\d+/.*?/(?:conv|fc)\d*/.*?Conv2D)"

                # Set to store the matching layer names
                matching_layer_names = set()

                # Traverse the graph to find operations matching the pattern
                for op in self.graph.get_operations():
                    match = re.search(pattern, op.name)
                    if match:
                        matching_layer_names.add(match.group(1))
                matching_layer_names = {
                    layer_name
                    for layer_name in matching_layer_names
                    if "gradients" not in layer_name
                }

                for tensor_name in matching_layer_names:
                    tensor_shape = self.graph.get_tensor_by_name(
                        f"{tensor_name}:0"
                    )
                    layer_shape_dict[tensor_name] = tensor_shape.shape

                for layer, shape in layer_shape_dict.items():
                    print(f"Layer: {layer}, Shape: {shape}")
                self.layer_shape_dict = layer_shape_dict

    def perform_forward_pass(
        self, output_tensor_name="tower_0/alexnet_v2/fc8/Conv2D:0"
    ):
        # Generate a random input tensor with the correct shape
        input_tensor_shape = [1] + self.image_shape  # Batch size of 1
        random_input = np.random.rand(*input_tensor_shape).astype(np.float32)

        # Perform a forward pass through the model
        with tf.Graph().as_default() as graph:
            with tf.compat.v1.Session(graph=graph) as sess:
                # Load the model
                tf.import_graph_def(self.graph_def, name="")
                input_tensor = graph.get_tensor_by_name(f"{self.input_name}:0")
                output_tensor = graph.get_tensor_by_name(
                    output_tensor_name
                )  # Example output tensor

                # Run the session to get the output
                output = sess.run(
                    output_tensor, feed_dict={input_tensor: random_input}
                )
                print("Output shape:", output.shape)
                print("Output values:", output)
