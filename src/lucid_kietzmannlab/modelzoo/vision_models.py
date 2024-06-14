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
from tqdm import tqdm

import lucid_kietzmannlab.optvis.objectives as objectives
import lucid_kietzmannlab.optvis.render as render
from lucid_kietzmannlab.misc.io import loading, showing
from lucid_kietzmannlab.modelzoo.conv_net import ConvNet
from lucid_kietzmannlab.modelzoo.vision_base import Model

tf.compat.v1.disable_eager_execution()
trunc_normal = lambda stddev: tf.compat.v1.truncated_normal_initializer(
    0.0, stddev
)

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
    sess,
    model,
    tensors_to_plot,
    input_placeholder,
    input_data,
    channels_first=False,
    feed_dict={},
):
    # Run the model to get the values of the specified tensors
    feed_dict[input_placeholder] = input_data
    layer_values_dict = {}
    for tensor_name in tensors_to_plot:
        print(model.graph.get_tensor_by_name(f"{tensor_name}:0").shape)
    tensor_values = sess.run(
        [
            model.graph.get_tensor_by_name(f"{tensor_name}:0")
            for tensor_name in tensors_to_plot
        ],
        feed_dict=feed_dict,
    )

    for tensor_name, tensor_value in zip(tensors_to_plot, tensor_values):
        layer_values_dict[tensor_name] = tensor_value
        plot_tensor_value = tensor_value[0, :]

        if channels_first:
            plot_tensor_value = np.transpose(plot_tensor_value, (1, 2, 0))
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


def plot_selected_layer_tensors(
    model, input_data, tensors_to_plot=[], channels_first=False
):
    with tf.Graph().as_default() as graph:
        with tf.compat.v1.Session(graph=graph) as sess:
            model.graph = sess.graph
            tf.import_graph_def(model.graph_def, name="")

            # Assuming input placeholder name is "input"

            input_placeholder = model.graph.get_tensor_by_name("Placeholder:0")
            try:
                boolean_placeholder = model.graph.get_tensor_by_name(
                    "load_data/Placeholder:0"
                )

                feed_dict = {boolean_placeholder: False}
            except Exception:

                feed_dict = {}

            model_tensor_plot(
                sess,
                model,
                tensors_to_plot,
                input_placeholder,
                input_data,
                channels_first=channels_first,
                feed_dict=feed_dict,
            )


def _get_layer_names_tensors(model: Model, scope=""):

    layer_name_list = [layer_info["name"] for layer_info in model.layers]
    t_input = tf.compat.v1.placeholder(tf.float32, [None, *model.image_shape])
    tf.compat.v1.import_graph_def(
        model.graph_def, {model.input_name: t_input}, name=scope
    )
    graph = tf.compat.v1.get_default_graph()

    layer_shape_dict = {}
    for layer_name in layer_name_list:
        try:
            tensor_shape = graph.get_tensor_by_name(f"{layer_name}:0").shape
            layer_shape_dict[layer_name] = tensor_shape
        except Exception:
            pass

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


class AlexNetCore(ConvNet):
    """
    Class used for building model.

    Attributes:
        output_size:     the number of classes to output with the readout
        keep_prob:       keep probability for dropout, set as a placeholder to vary from training to
                         validation
    """

    def __init__(
        self,
        input_images,
        reuse_variables=False,
        n_layers=7,
        default_timesteps=1,
        data_format="NCHW",
        var_device="/cpu:0",
        model_name="b_net",
        random_seed=None,
    ):
        """
        Args:
            input_images:      input images to the network should be 4-d for images,
                               e.g. [batch, channels, height, width], or 5-d for movies,
                               e.g. [batch, time, channels, height, width]
            reuse_variables:   whether to create or reuse variables
            data_format:       NCHW for GPUs and NHWC for CPUs
            var_device:        device for storing model variables (recognisable by tensorflow),
                               CPU is recommended when parallelising across GPUs, GPU is recommended
                               when a single GPU is used
        """

        default_timesteps = 1
        ConvNet.__init__(
            self,
            input_images,
            reuse_variables,
            n_layers,
            default_timesteps,
            data_format,
            var_device,
            model_name,
            random_seed,
        )

        # the only things that matter
        self.output_size = 2  # number of classes
        self.activations = [
            [None] * self.n_layers for _ in range(self.n_timesteps)
        ]
        self.readout = [None] * self.n_timesteps
        self.keep_prob = 0.5

        # set in the script but ignored
        self.dropout_type = None
        self.is_training = None

    def get_model_params(self, affix="model_params", ignore_attr=[]):
        """
        Returns a dictionary containing parmeters defining the model ignoring any methods and
        any attributes in self.get_params_ignore or ignore_attr
        """

        model_param_dict = {"ALEXNET": "special case"}

        return model_param_dict

    def build_model(self):
        """
        Builds the computational graph for the model
        """

        with slim.arg_scope(
            [slim.model_variable, slim.variable], device=self.var_device
        ):
            readout, act, weights = alexnet_v2(
                self.input,
                self.output_size,
                dropout_keep_prob=self.keep_prob,
                reuse_variables=self.reuse_variables,
                data_format=self.data_format,
            )

        self.readout = [readout]
        self.weight_list = weights
        self.activations = [act]


class AlexNetCodeOcean(Model):

    def __init__(
        self,
        model_dir,
        random_seed=1,
        output_size=565,
        vis_layer=1,
        channel=0,
        thresholds=(512,),
    ):

        self.ckpt_path = os.path.join(model_dir, "model.ckpt_epoch89")
        self.model_dir = model_dir
        self.output_size = output_size
        self.image_shape = [224, 224, 3]
        self.dataset = "Ecoset"
        self._vis_layer = vis_layer
        self._channel = channel
        self.is_BGR = False
        self.image_value_range = (-IMAGENET_MEAN_BGR, 255 - IMAGENET_MEAN_BGR)
        self.input_name = "image_ph"
        self.dropout_type = "bernoulli"
        self.random_seed = random_seed
        self.thresholds = thresholds
        shape = [1, *self.image_shape]
        loaded_images = np.random.rand(*shape)
        self._load_model_layers()
        self.graph = self._build_graph(loaded_images)
        self.layer_shape_dict = _get_layer_names_tensors(self)

    @property
    def vis_layer(self):
        return self._vis_layer

    @vis_layer.setter
    def vis_layer(self, value):
        self._vis_layer = value

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, value):
        self._channel = value

    @property
    def graph_def(self):
        if not self._graph_def:
            self._graph_def = self.graph.as_graph_def()
        return self._graph_def

    def lucid_visualize_layer(
        self, batch=False, channel_start=0, channel_end=-1
    ):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        C = lambda neuron: objectives.channel(*neuron)
        shape = [1, *self.image_shape]
        loaded_images = np.random.rand(*shape)
        self.graph = self._build_graph(loaded_images)

        with tf.compat.v1.Session(graph=self.graph, config=config) as sess:
            self.activations = self.model.activations
            self.activations_readout_before_softmax = self.model.readout
            self.activations_readout_after_softmax = tf.nn.softmax(
                self.activations_readout_before_softmax
            )
            saver = tf.compat.v1.train.Saver()
            saver.restore(sess, self.ckpt_path)

            if isinstance(self._vis_layer, int):
                layer_name = self.layers[self._vis_layer]["name"]
            else:
                layer_name = self._vis_layer
            channel = self._channel

            tensor = self.graph.get_tensor_by_name(f"{layer_name}:0")
            tensor_shape = tensor.shape
            max_channel = tensor_shape[-1] - 1
            image_channel = {}
            if batch:
                if channel_end == -1:
                    channel_end = max_channel
                for channel in tqdm(range(channel_start, channel_end)):
                    objective_f = C((layer_name, channel))
                    T = render.make_vis_T(
                        self, objective_f, scope=f"import_{channel}"
                    )

                    loss, vis_op, t_image = T("loss"), T("vis_op"), T("input")
                    tf.compat.v1.global_variables_initializer().run()
                    images = []

                    try:
                        for i in range(max(self.thresholds) + 1):
                            loss_, _ = sess.run([loss, vis_op])
                            if i in self.thresholds:
                                vis = t_image.eval()
                                images.append(vis)
                        image_channel[channel] = images
                    except KeyboardInterrupt:
                        vis = t_image.eval()
                        showing.show(np.hstack(vis))
            else:
                objective_f = C((layer_name, channel))
                T = render.make_vis_T(
                    self,
                    objective_f,
                )

                loss, vis_op, t_image = T("loss"), T("vis_op"), T("input")
                tf.compat.v1.global_variables_initializer().run()
                images = []

                try:
                    for i in range(max(self.thresholds) + 1):
                        loss_, _ = sess.run([loss, vis_op])
                        if i in self.thresholds:
                            vis = t_image.eval()
                            images.append(vis)
                            showing.show(np.hstack(vis))
                    image_channel[channel] = images
                except KeyboardInterrupt:
                    vis = t_image.eval()
                    showing.show(np.hstack(vis))

            return image_channel

    def _build_graph(
        self,
        loaded_images,
        n_timesteps=1,
        output_size=565,
    ):

        graph = tf.Graph()
        with graph.as_default():

            if self.random_seed is not None:
                tf.compat.v1.set_random_seed(self.random_seed)

            model_device = "/cpu:0"
            data_format = "NHWC"
            print("loaded_images", loaded_images.shape)
            img_ph = tf.compat.v1.placeholder(
                tf.float32, np.shape(loaded_images), self.input_name
            )
            image_float32 = tf.image.convert_image_dtype(img_ph, tf.float32)
            image_rescaled = (image_float32 - 0.5) * 2

            image_tiled = tf.tile(
                tf.expand_dims(image_rescaled, 0), [1, 1, 1, 1, 1]
            )
            self.model = AlexNetCore(
                image_tiled,
                var_device=model_device,
                default_timesteps=n_timesteps,
                data_format=data_format,
                random_seed=self.random_seed,
            )

            self.model.output_size = output_size
            self.model.dropout_type = self.dropout_type
            self.model.net_mode = "test"
            self.model.is_training = False
            self.model.build_model()

        return graph

    def _load_model_layers(self):
        self.layers = _layers_from_list_of_dicts(
            self,
            [
                {
                    "tags": ["conv"],
                    "name": "alexnet_v2/conv1/Conv2D",
                    "depth": 64,
                },
                {
                    "tags": ["conv"],
                    "name": "alexnet_v2/conv2/Conv2D",
                    "depth": 192,
                },
                {
                    "tags": ["conv"],
                    "name": "alexnet_v2/conv3/Conv2D",
                    "depth": 384,
                },
                {
                    "tags": ["conv"],
                    "name": "alexnet_v2/conv4/Conv2D",
                    "depth": 384,
                },
                {
                    "tags": ["conv"],
                    "name": "alexnet_v2/conv5/Conv2D",
                    "depth": 256,
                },
                {
                    "tags": ["dense"],
                    "name": "alexnet_v2/fc6/Conv2D",
                    "depth": 4096,
                },
                {
                    "tags": ["dense"],
                    "name": "alexnet_v2/fc7/Conv2D",
                    "depth": 4096,
                },
                {
                    "tags": ["dense"],
                    "name": "alexnet_v2/fc8/Conv2D",
                    "depth": 565,
                },
            ],
        )


class AlexNetv2(Model):

    def __init__(
        self,
        model_checkpoint_dir="/Users/vkapoor/Downloads/models/AlexNet/training_seed_05",
        training_seed=5,
        scope="",
    ):
        self.image_shape = [3, 227, 227]
        self.dataset = "Ecoset"
        self.is_BGR = False
        self.image_value_range = (-IMAGENET_MEAN_BGR, 255 - IMAGENET_MEAN_BGR)
        self.input_name = "Placeholder"
        self.model_checkpoint_dir = model_checkpoint_dir
        self.training_seed = training_seed
        self.scope = scope
        self.model_name = os.path.join(
            self.model_checkpoint_dir, "model.ckpt_epoch89"
        )

        print(f"Loading model with name {self.model_name}")
        self.meta_path = os.path.join(
            model_checkpoint_dir, "model.ckpt_epoch89.meta"
        )
        self.load_model_layers()
        self.layer_shape_dict = _get_layer_names_tensors(self)

    @property
    def graph_def(self):
        if not self._graph_def:
            with tf.compat.v1.Session() as sess:
                saver = tf.compat.v1.train.import_meta_graph(
                    self.meta_path, clear_devices=True
                )
                saver.restore(sess, self.model_name)
                graph = tf.compat.v1.get_default_graph()

                # Initialize all variables
                sess.run(tf.compat.v1.global_variables_initializer())

                self._graph_def = graph.as_graph_def()

        return self._graph_def

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
            ],
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


def get_weights():
    return [
        v
        for v in tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES
        )
        if v.name.endswith("weights:0")
    ]


def alexnet_v2(
    inputs,
    num_classes=1000,
    is_training=True,
    dropout_keep_prob=0.5,
    spatial_squeeze=True,
    scope="alexnet_v2",
    global_pool=False,
    reuse_variables=True,
    data_format="NHWC",
):
    """AlexNet version 2.
    Described in: http://arxiv.org/pdf/1404.5997v2.pdf
    Parameters from:
    github.com/akrizhevsky/cuda-convnet2/blob/master/layers/
    layers-imagenet-1gpu.cfg
    Note: All the fully_connected layers have been transformed to conv2d layers.
          To use in classification mode, resize input to 224x224 or set
          global_pool=True. To use in fully convolutional mode, set
          spatial_squeeze to false.
          The LRN layers have been removed and change the initializers from
          random_normal_initializer to xavier_initializer.
    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      num_classes: the number of predicted classes. If 0 or None, the logits layer
      is omitted and the input features to the logits layer are returned instead.
      is_training: whether or not the model is being trained.
      dropout_keep_prob: the probability that activations are kept in the dropout
        layers during training.
      spatial_squeeze: whether or not should squeeze the spatial dimensions of the
        logits. Useful to remove unnecessary dimensions for classification.
      scope: Optional scope for the variables.
      global_pool: Optional boolean flag. If True, the input to the classification
        layer is avgpooled to size 1x1, for any input size. (This is not part
        of the original AlexNet.)
    Returns:
      net: the output of the logits layer (if num_classes is a non-zero integer),
        or the non-dropped-out input to the logits layer (if num_classes is 0
        or None).
      end_points: a dict of tensors with intermediate activations.
    """
    act = []
    with tf.compat.v1.variable_scope(
        scope, "alexnet_v2", [inputs], reuse=reuse_variables
    ) as sc:
        end_points_collection = sc.original_name_scope + "_end_points"
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope(
            [slim.conv2d, slim.fully_connected, slim.max_pool2d],
            outputs_collections=[end_points_collection],
            data_format=data_format,
        ):
            inputs = inputs[0, :]
            net = slim.conv2d(
                inputs, 64, [11, 11], 4, padding="VALID", scope="conv1"
            )
            act.append(net)
            net = slim.max_pool2d(net, [3, 3], 2, scope="pool1")
            net = slim.conv2d(net, 192, [5, 5], scope="conv2")
            act.append(net)
            net = slim.max_pool2d(net, [3, 3], 2, scope="pool2")
            net = slim.conv2d(net, 384, [3, 3], scope="conv3")
            act.append(net)
            net = slim.conv2d(net, 384, [3, 3], scope="conv4")
            act.append(net)
            net = slim.conv2d(net, 256, [3, 3], scope="conv5")
            act.append(net)
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
                act.append(net)
                net = slim.conv2d(net, 4096, [1, 1], scope="fc7")
                act.append(net)
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
                        squeeze_dims = (
                            [2, 3] if data_format == "NCHW" else [1, 2]
                        )
                        net = tf.squeeze(
                            net, squeeze_dims, name="fc8/squeezed"
                        )
                    readout = net
                    end_points[sc.name + "/fc8"] = net
            return readout, act, get_weights()


alexnet_v2.default_image_size = 224
