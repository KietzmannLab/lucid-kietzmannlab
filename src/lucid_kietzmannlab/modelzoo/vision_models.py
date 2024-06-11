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
import warnings

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tf_slim as slim
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


def model_tensor_plot(
    sess, model, tensors_to_plot, input_placeholder, input_data
):
    # Run the model to get the values of the specified tensors
    layer_values_dict = {}
    tensor_values = sess.run(
        [
            model.graph.get_tensor_by_name(f"{tensor_name}:0")
            for tensor_name in tensors_to_plot
        ],
        feed_dict={input_placeholder: input_data},
    )

    for tensor_name, tensor_value in zip(tensors_to_plot, tensor_values):
        layer_values_dict[tensor_name] = tensor_value
        plot_tensor_value = tensor_value[0, :]
        num_channels = plot_tensor_value.shape[-1]
        # Calculate grid size
        if len(plot_tensor_value.shape) > 1:
            grid_size = int(np.ceil(np.sqrt(num_channels)))

            # Plot images in a grid
            plt.figure(figsize=(15, 15))
            for i in range(num_channels):
                plt.subplot(grid_size, grid_size, i + 1)
                plt.imshow(plot_tensor_value[:, :, i], cmap="gray")
                plt.axis("off")
            plt.suptitle(f"{tensor_name} - Images")
            plt.show()
        else:

            plt.plot(plot_tensor_value)
            plt.suptitle(f"{tensor_name}")
            plt.show()


def plot_selected_layer_tensors(model, input_data, tensors_to_plot=[]):
    if isinstance(model, AlexNet):
        with tf.Graph().as_default() as graph:
            with tf.compat.v1.Session(graph=graph) as sess:
                model.graph = sess.graph
                tf.import_graph_def(model.graph_def, name="")

                # Assuming input placeholder name is "input"

                input_placeholder = model.graph.get_tensor_by_name(
                    "Placeholder:0"
                )
                model_tensor_plot(
                    sess, model, tensors_to_plot, input_placeholder, input_data
                )


def _get_layer_names_tensors(model):

    layer_name_list = [layer_info["name"] for layer_info in model.layers]
    # Get the shape of each layer
    with tf.Graph().as_default() as graph:
        # Import the model
        tf.import_graph_def(model.graph_def, name="")

        # Get the shape of each tensor
        layer_shape_dict = {}
        with tf.compat.v1.Session() as sess:
            for layer_name in layer_name_list:
                try:
                    tensor_shape = sess.graph.get_tensor_by_name(
                        f"{layer_name}:0"
                    ).shape

                    if tensor_shape != ():

                        if tensor_shape[0] is None:

                            layer_shape_dict[layer_name] = tensor_shape
                except KeyError:
                    # Handle the case where the tensor is not found
                    layer_shape_dict[layer_name] = None

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
    image_shape = [227, 227, 3]
    is_BGR = True
    image_value_range = (-IMAGENET_MEAN_BGR, 255 - IMAGENET_MEAN_BGR)
    input_name = "Placeholder"

    def __init__(self):
        self.layers = _layers_from_list_of_dicts(
            self,
            [
                {"tags": ["pre_relu", "conv"], "name": "Conv2D", "depth": 96},
                {
                    "tags": ["pre_relu", "conv"],
                    "name": "Conv2D_1",
                    "depth": 128,
                },
                {
                    "tags": ["pre_relu", "conv"],
                    "name": "Conv2D_2",
                    "depth": 128,
                },
                {
                    "tags": ["pre_relu", "conv"],
                    "name": "Conv2D_3",
                    "depth": 384,
                },
                {
                    "tags": ["pre_relu", "conv"],
                    "name": "Conv2D_4",
                    "depth": 192,
                },
                {
                    "tags": ["pre_relu", "conv"],
                    "name": "Conv2D_5",
                    "depth": 192,
                },
                {
                    "tags": ["pre_relu", "conv"],
                    "name": "Conv2D_6",
                    "depth": 128,
                },
                {
                    "tags": ["pre_relu", "conv"],
                    "name": "Conv2D_7",
                    "depth": 128,
                },
                {"tags": ["dense"], "name": "Relu", "depth": 4096},
                {"tags": ["dense"], "name": "Relu_1", "depth": 4096},
                {"tags": ["dense"], "name": "Softmax", "depth": 1000},
            ],
        )
        self.layer_shape_dict = _get_layer_names_tensors(self)


class AlexNetv2(Model):

    def __init__(
        self,
        model_checkpoint_dir="/Users/vkapoor/Downloads/models/AlexNet/training_seed_05",
        training_seed=5,
    ):
        self.image_shape = [224, 224, 3]
        self.dataset = "Ecoset"
        self.is_BGR = False
        self.image_value_range = (-IMAGENET_MEAN_BGR, 255 - IMAGENET_MEAN_BGR)
        self.input_name = "Placeholder"
        self.model_checkpoint_dir = model_checkpoint_dir
        self.training_seed = training_seed
        self.model_name = os.path.join(
            self.model_checkpoint_dir, "model.ckpt_epoch89"
        )
        self.meta_path = os.path.join(
            model_checkpoint_dir, "model.ckpt_epoch89.meta"
        )

        self.layer_shape_dict = _get_layer_names_tensors(self)

    @property
    def graph_def(self):
        if not self._graph_def:
            with tf.compat.v1.Session() as sess:
                saver = tf.compat.v1.train.import_meta_graph(
                    self.meta_path, clear_devices=True
                )
                saver.restore(sess, self.model_name)

                output_node_names = [
                    n.name
                    for n in tf.compat.v1.get_default_graph()
                    .as_graph_def()
                    .node
                ]
                frozen_graph_def = (
                    tf.compat.v1.graph_util.convert_variables_to_constants(
                        sess, sess.graph_def, output_node_names
                    )
                )

            self._graph_def = frozen_graph_def
        return self._graph_def

    def load_model_layers(self):
        self.layers = _layers_from_list_of_dicts(
            self,
            [
                {"tags": ["conv"], "name": "conv1/Conv2D", "depth": 64},
                {"tags": ["conv"], "name": "conv2/Conv2D", "depth": 192},
                {"tags": ["conv"], "name": "conv3/Conv2D", "depth": 384},
                {"tags": ["conv"], "name": "conv4/Conv2D", "depth": 384},
                {"tags": ["conv"], "name": "conv5/Conv2D", "depth": 256},
                {"tags": ["dense"], "name": "fc6/Conv2D", "depth": 4096},
                {"tags": ["dense"], "name": "fc7/Conv2D", "depth": 4096},
                {"tags": ["dense"], "name": "fc8/Conv2D", "depth": 565},
            ],
        )


trunc_normal = lambda stddev: tf.compat.v1.truncated_normal_initializer(
    0.0, stddev
)


def alexnet_v2_arg_scope(weight_decay=0.0005):
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        activation_fn=tf.nn.relu,
        biases_initializer=tf.constant_initializer(0.1),
        weights_regularizer=slim.l2_regularizer(weight_decay),
    ):
        with slim.arg_scope([slim.conv2d], padding="SAME"):
            with slim.arg_scope([slim.max_pool2d], padding="VALID") as arg_sc:
                return arg_sc


def alexnet_v2(
    inputs,
    num_classes=1000,
    is_training=True,
    dropout_keep_prob=0.5,
    spatial_squeeze=True,
    reuse_variables=tf.compat.v1.AUTO_REUSE,
    scope="alexnet_v2",
    global_pool=False,
):

    with tf.compat.v1.variable_scope(
        scope, "alexnet_v2", [inputs], reuse=reuse_variables
    ) as sc:
        end_points_collection = sc.original_name_scope + "_end_points"
        with slim.arg_scope(
            [slim.conv2d, slim.fully_connected, slim.max_pool2d],
            outputs_collections=end_points_collection,
        ):
            net = slim.conv2d(
                inputs, 64, [11, 11], 4, padding="VALID", scope="conv1"
            )
            net = slim.max_pool2d(net, [3, 3], 2, scope="pool1")
            net = slim.conv2d(net, 192, [5, 5], scope="conv2")
            net = slim.max_pool2d(net, [3, 3], 2, scope="pool2")
            net = slim.conv2d(net, 384, [3, 3], scope="conv3")
            net = slim.conv2d(net, 384, [3, 3], scope="conv4")
            net = slim.conv2d(net, 256, [3, 3], scope="conv5")
            net = slim.max_pool2d(net, [3, 3], 2, scope="pool5")

            # Use conv2d instead of fully_connected layers.
            with slim.arg_scope(
                [slim.conv2d],
                weights_initializer=trunc_normal(0.005),
                biases_initializer=tf.constant_initializer(0.1),
            ):
                net = slim.conv2d(
                    net, 4096, [5, 5], padding="VALID", scope="fc6"
                )
                net = slim.dropout(
                    net,
                    dropout_keep_prob,
                    is_training=is_training,
                    scope="dropout6",
                )
                net = slim.conv2d(net, 4096, [1, 1], scope="fc7")
                # Convert end_points_collection into a end_point dict.
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection
                )
                if global_pool:
                    net = tf.reduce_mean(
                        input_tensor=net,
                        axis=[1, 2],
                        keepdims=True,
                        name="global_pool",
                    )
                    end_points["global_pool"] = net
                if num_classes:
                    net = slim.dropout(
                        net,
                        dropout_keep_prob,
                        is_training=is_training,
                        scope="dropout7",
                    )
                    net = slim.conv2d(
                        net,
                        num_classes,
                        [1, 1],
                        activation_fn=None,
                        normalizer_fn=None,
                        biases_initializer=tf.zeros_initializer(),
                        scope="fc8",
                    )
                if spatial_squeeze:
                    net = tf.squeeze(net, [1, 2], name="fc8/squeezed")
                    end_points[sc.name + "/fc8"] = net
    return net, end_points


alexnet_v2.default_image_size = 224
