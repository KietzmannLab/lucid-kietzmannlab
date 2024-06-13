"""
Input arguments
sys.argv[1]: model_folder_file_path excluding filename ending
sys.argv[2]: output_size
sys.argv[3]: 1 - model.keep_prob
sys.argv[4]: training_seed
sys.argv[5]: DISTANCE_MEASURE
sys.argv[6]: save_activations
sys.argv[7]: training_set (ecoset vs. ILSVRC2012)
sys.argv[8]: layer
sys.argv[9]: compute_RDMs_bool
"""

import glob
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf
import tf_slim as slim
from PIL import Image
from scipy.io import loadmat, savemat
from scipy.spatial.distance import pdist

from lucid_kietzmannlab.modelzoo.conv_net import ConvNet

tf.compat.v1.disable_eager_execution()
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


class AlexNet(ConvNet):
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


CKPT_PATH = sys.argv[1]  # CHANGE 06
output_size = sys.argv[2]
dropout_rate = 0.5
random_seed = sys.argv[3]
DISTANCE_MEASURE = sys.argv[4]
IMAGE_DIR = sys.argv[5]
training_set = sys.argv[6]
layer = int(sys.argv[7])
compute_RDMs_bool = int(sys.argv[8])

np.random.seed(int(sys.argv[3]))
# sets the graph based random seed
RANDOM_SEED_GRAPH = int(sys.argv[3])


argv_1 = CKPT_PATH
argv_1_list = argv_1.split(os.sep)
ckpt_name = argv_1_list[-1]
save_name = f"training_seed_{random_seed.zfill(2)}"  # argv_1_list[len(argv_1_list) - 2]
# dropout type
DROPOUT_TYPE = "bernoulli"

# path to the specific checkpoint file
ckpt_path = CKPT_PATH

# path to the model that we feed the images to
model_folder = __file__

sys.path.append("/data/models/AlexNet/")


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# results_dir: every output from the run of a codeocean capsule is saved in the folder "results" in the main folder
results_dir = "../results"
save_dir = os.path.join(results_dir, save_name)

# folder containing all RDL92/61 categories
exp_stimuli_path = IMAGE_DIR  # image_input_only_niko_92, kamitani


# %% model specs
model_type = "b_net_alexnet"
scale_blt = [1.0, 1.0, 1.0]
scale_parameters = None
n_layers = 7
n_timesteps = 1
layer_features = (64, 192, 384, 384, 256, 4096, 4096)
k_size = (11, 5, 3, 3, 3, 3, None, None)
pool_size = (None, 3, 3, None, 3, None, None)
output_size = int(sys.argv[2])  # 565 vs 1000
save_network_activations = True
rdm_times = 0


# %% load stimuli


def load_images(exp_stimuli_path):
    """
    Loads images resize to 224x224

    Returns:
        image_list:         stack of loaded images in 224x224 resolution
        image_name_list:    list of names of loaded files for saving of mat files

    """
    # get all files in the directory and sort them
    dir_files = sorted(glob.glob(os.path.join(exp_stimuli_path, "*")))

    # initialise the list to store images
    images_list = []

    # keep track of ignored files in folder
    ignored_files = 0

    image_name_list = []
    for fname in dir_files:

        # get file name for later saving of activations
        img_file_name = os.path.splitext(os.path.basename(fname))[0]
        image_name_list.append(img_file_name)

        # load the image checking if it is a valid file type
        try:
            img_i = Image.open(fname)

        except OSError:
            print(f"not a recognised image file, ignoring: {fname}")
            ignored_files += 1
            continue

        # HPC HPC HPC crop center part of image
        width, height = img_i.size
        min_dim = min([width, height])
        y_crop = min_dim / width
        x_crop = min_dim / height
        y0 = (1 - y_crop) / 2
        x0 = (1 - x_crop) / 2
        y1 = 1 - y0
        x1 = 1 - x0
        crop_box = [y0, x0, y1, x1]
        img_i_array_cropped = img_i.crop(
            (
                round(crop_box[0] * width),
                round(crop_box[1] * height),
                round(crop_box[2] * width),
                round(crop_box[3] * height),
            )
        )

        # HPC HPC HPC resize image to fit input size of DNN
        img_i_array_resized_tmp = img_i_array_cropped.resize(
            (224, 224), Image.BILINEAR
        )
        img_i_array_resized = np.asarray(
            img_i_array_resized_tmp
        )  # HPC HPC HPC
        images_list.append(img_i_array_resized)

    return np.array(images_list), image_name_list


def build_graph(
    loaded_images,
    n_timesteps=1,
    output_size=565,
):

    graph = tf.Graph()
    with graph.as_default():

        if RANDOM_SEED_GRAPH is not None:
            tf.compat.v1.set_random_seed(RANDOM_SEED_GRAPH)

        model_device = "/cpu:0"
        data_format = "NHWC"
        print(loaded_images.shape)
        img_ph = tf.compat.v1.placeholder(
            tf.float32, np.shape(loaded_images)[1:], "images_ph"
        )
        image_float32 = tf.image.convert_image_dtype(img_ph, tf.float32)
        image_rescaled = (image_float32 - 0.5) * 2
        images_tensor = image_rescaled

        image_tiled = tf.tile(tf.expand_dims(image_rescaled, 0), [1, 1, 1, 1])
        model = AlexNet(
            image_tiled,
            var_device=model_device,
            default_timesteps=n_timesteps,
            data_format=data_format,
            random_seed=RANDOM_SEED_GRAPH,
        )

        model.output_size = output_size
        model.dropout_type = DROPOUT_TYPE
        model.net_mode = "test"
        model.is_training = False
        model.build_model()

    return graph, images_tensor, model, img_ph


def extract_and_save_activations(
    loaded_images,
    image_name_list,
    save_path_activations,
    graph,
    images_tensor,
    model,
    ckpt_path,
    exp_stimuli_cat_limit,
    layer,
    img_ph,
):
    """
    Extract the responses of the network to images

    Args:
        images:       an array of images of shape [n_images, width, height, channels]
        graph:        the graph object for the network returned from build_graph
        image_tensor: placeholder corresponding to the image tensor
        model:        the model object for the network returned from build_graph
        ckpt_path:    the path to the checkpoint file to restore from
        cat:                           category for checking amount of images per category

    Returns:
        act_list:    a list containing the activations for each layer to all images in the following format [time, image, channels, height, width]

    """

    n_images = loaded_images.shape[0]
    if n_images != exp_stimuli_cat_limit:
        print(str(n_images) + " does not equal " + str(exp_stimuli_cat_limit))
        os._exit()

    # set-up config object for session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.compat.v1.Session(graph=graph, config=config) as sess:

        activations = model.activations
        activations_readout_before_softmax = (
            model.readout
        )  # batch_size x class
        activations_readout_after_softmax = tf.nn.softmax(
            activations_readout_before_softmax
        )

        # restore the network
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, ckpt_path)

        # progress bar
        print("extracting activations...")

        for image_i, image in enumerate(loaded_images):
            start = time.time()
            # transpose and reshape image
            input_image = image  # CPU-mode: don't transpose or change shape
            # set random seeds
            np.random.seed(int(sys.argv[3]))
            tf.compat.v1.set_random_seed(int(sys.argv[3]))

            ops_to_run = [
                activations,
                activations_readout_before_softmax,
                activations_readout_after_softmax,
            ]

            (
                batch_act,
                batch_act_readout_before_softmax,
                batch_act_readout_after_softmax,
            ) = sess.run(ops_to_run, feed_dict={img_ph: input_image})

            # save activations
            activation_dict = {}
            for layer_i in [layer]:  # CHANGE 15
                if layer_i == 0:
                    activation_dict[f"layer_{str(layer_i + 1).zfill(3)}"] = (
                        np.squeeze(batch_act[0][layer_i])
                    )
                    print("layer 0")
                    print(
                        activation_dict[
                            f"layer_{str(layer_i + 1).zfill(3)}"
                        ].shape
                    )
                    savemat(
                        os.path.join(
                            save_path_activations,
                            f"layer_{str(layer_i + 1).zfill(3)}_img_idx_{str(image_i + 1).zfill(4)}_image_id_{image_name_list[image_i]}_random_seed_{str(int(sys.argv[3])).zfill(2)}_training_set_{training_set}.mat",
                        ),
                        activation_dict,
                    )
                    activation_dict = {}

                # layer 2-5 - all conv layers, but the first
                elif layer_i < 5:
                    activation_dict[f"layer_{str(layer_i + 1).zfill(3)}"] = (
                        np.squeeze(batch_act[0][layer_i])
                    )
                    print("layer 2-5")
                    print(
                        activation_dict[
                            f"layer_{str(layer_i + 1).zfill(3)}"
                        ].shape
                    )
                    savemat(
                        os.path.join(
                            save_path_activations,
                            f"layer_{str(layer_i + 1).zfill(3)}_img_idx_{str(image_i + 1).zfill(4)}_image_id_{image_name_list[image_i]}_random_seed_{str(int(sys.argv[3])).zfill(2)}_training_set_{training_set}.mat",
                        ),
                        activation_dict,
                    )
                    activation_dict = {}

                # the fully connected layers
                elif layer_i == 5 or layer_i == 6:
                    activation_dict[f"layer_{str(layer_i + 1).zfill(3)}"] = (
                        np.squeeze(batch_act[0][layer_i])
                    )
                    print("layer 6-7")
                    print(
                        activation_dict[
                            f"layer_{str(layer_i + 1).zfill(3)}"
                        ].shape
                    )
                    savemat(
                        os.path.join(
                            save_path_activations,
                            f"layer_{str(layer_i + 1).zfill(3)}_img_idx_{str(image_i + 1).zfill(4)}_image_id_{image_name_list[image_i]}_random_seed_{str(int(sys.argv[3])).zfill(2)}_training_set_{training_set}.mat",
                        ),
                        activation_dict,
                    )
                    activation_dict = {}

                # readout before softmax
                elif layer_i == model.n_layers:
                    activation_dict[f"layer_{str(layer_i + 1).zfill(3)}"] = (
                        np.squeeze(batch_act_readout_before_softmax)
                    )
                    print("readout before softmax")
                    print(
                        len(
                            activation_dict[
                                f"layer_{str(layer_i + 1).zfill(3)}"
                            ]
                        )
                    )
                    savemat(
                        os.path.join(
                            save_path_activations,
                            f"layer_{str(layer_i + 1).zfill(3)}_img_idx_{str(image_i + 1).zfill(4)}_image_id_{image_name_list[image_i]}_random_seed_{str(int(sys.argv[3])).zfill(2)}_training_set_{training_set}.mat",
                        ),
                        activation_dict,
                    )
                    activation_dict = {}

                # readout after softmax
                elif layer_i == model.n_layers + 1:
                    activation_dict[f"layer_{str(layer_i + 1).zfill(3)}"] = (
                        np.squeeze(batch_act_readout_after_softmax)
                    )
                    print("readout after softmax")
                    print(
                        len(
                            activation_dict[
                                f"layer_{str(layer_i + 1).zfill(3)}"
                            ]
                        )
                    )
                    savemat(
                        os.path.join(
                            save_path_activations,
                            f"layer_{str(layer_i + 1).zfill(3)}_img_idx_{str(image_i + 1).zfill(4)}_image_id_{image_name_list[image_i]}_random_seed_{str(int(sys.argv[3])).zfill(2)}_training_set_{training_set}.mat",
                        ),
                        activation_dict,
                    )
                    activation_dict = {}
            end = time.time()


# %% compute RDMs
def compute_RDMs(
    loaded_images,
    image_name_list,
    save_path_activations,
    graph,
    images_tensor,
    model,
    ckpt_path,
    exp_stimuli_cat_limit,
    layer,
):

    save_name = f"training_seed_{random_seed.zfill(2)}"
    cwd = os.getcwd()
    cwd_split = os.path.split(cwd)
    cwd_parent = cwd_split[0]
    results_dir = os.path.join(cwd_parent, "results")
    save_dir = os.path.join(results_dir, save_name)

    # get layer-specific file list
    IMAGE_DIR_split = os.path.split(IMAGE_DIR)
    IMAGE_DIR_split_2 = os.path.split(IMAGE_DIR_split[0])
    save_path_activations = os.path.join(
        save_dir, IMAGE_DIR_split_2[-1], "activations"
    )
    layer_activations_files = sorted(
        glob.glob(
            os.path.join(
                save_path_activations,
                f"layer_{str(int(layer) + 1).zfill(3)}*{training_set}*",
            )
        )
    )

    # load files
    layer_activations_tmp = []
    for img in layer_activations_files:
        tmp = loadmat(img)
        tmp_2 = np.ndarray.flatten(
            tmp[f"layer_{str(int(layer) + 1).zfill(3)}"]
        )
        layer_activations_tmp.append(tmp_2)
    layer_activations = np.vstack(layer_activations_tmp)

    # compute RDM
    RDM = pdist(layer_activations, DISTANCE_MEASURE)

    # create RDMs save dir
    save_path_RDMs = os.path.join(save_dir, IMAGE_DIR_split_2[-1], "RDMs")
    if not os.path.isdir(save_path_RDMs):
        os.mkdir(save_path_RDMs)

    # save RDM in .mat (matlab) and .npy (python numpy) format
    RDM_dict = {}
    RDM_dict["RDM"] = RDM
    savemat(
        os.path.join(
            save_path_RDMs,
            f"{save_name}_{cwd_split[1]}_layer_{str(int(layer) + 1).zfill(3)}_dst_measure_{DISTANCE_MEASURE}.mat",
        ),
        RDM_dict,
    )
    np.save(
        os.path.join(
            save_path_RDMs,
            f"{save_name}_{cwd_split[1]}_layer_{str(int(layer) + 1).zfill(3)}_dst_measure_{DISTANCE_MEASURE}",
        ),
        RDM,
    )


# %% run script

# create activation save path parent folders
if not os.path.isdir(os.path.join(results_dir)):
    os.mkdir(os.path.join(results_dir))
if not os.path.isdir(os.path.join(save_dir)):
    os.mkdir(os.path.join(save_dir))

# create save path structure for each category
IMAGE_DIR_split = os.path.split(IMAGE_DIR)
IMAGE_DIR_split_2 = os.path.split(IMAGE_DIR_split[0])
save_path_activations_tmp = os.path.join(save_dir, IMAGE_DIR_split_2[-1])
if not os.path.isdir(save_path_activations_tmp):
    os.mkdir(save_path_activations_tmp)
save_path_activations = os.path.join(
    save_dir, IMAGE_DIR_split_2[-1], "activations"
)
if not os.path.isdir(save_path_activations):
    os.mkdir(save_path_activations)


# load experimental stimuli
tmp_cat_n_imgs = os.listdir(os.path.join(exp_stimuli_path))
exp_stimuli_cat_limit = len(tmp_cat_n_imgs)


loaded_images, image_name_list = load_images(os.path.join(exp_stimuli_path))


# build the tensor graph
graph, images_tensor, model, img_ph = build_graph(
    loaded_images,
    n_timesteps=n_timesteps,
    output_size=output_size,
)

extract_and_save_activations(
    loaded_images,
    image_name_list,
    save_path_activations,
    graph,
    images_tensor,
    model,
    ckpt_path,
    exp_stimuli_cat_limit,
    layer,
    img_ph,
)

if compute_RDMs_bool:
    compute_RDMs(
        loaded_images,
        image_name_list,
        save_path_activations,
        graph,
        images_tensor,
        model,
        ckpt_path,
        exp_stimuli_cat_limit,
        layer,
    )


# save analysis parameters
analysis_params = {
    "model_folder": model_folder,
    "ckpt_name": ckpt_name,
    "save_dir": save_name,
    "model_type": model_type,
    "n_layers": n_layers,
    "layer_features": layer_features,
}

with open(
    os.path.join(results_dir, save_dir, "analysis_params.json"), "w"
) as fp:
    json.dump(analysis_params, fp)
