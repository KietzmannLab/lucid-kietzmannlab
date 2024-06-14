import logging

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython.display import clear_output
from scipy.ndimage import convolve
from skimage.filters import gabor_kernel
from tqdm import tqdm

import lucid_kietzmannlab.optvis.objectives as objectives
import lucid_kietzmannlab.optvis.render as render
from lucid_kietzmannlab.misc.io import showing

# pylint: disable=invalid-name


# create logger with module name, e.g. lucid.misc.io.reading
log = logging.getLogger(__name__)


def plot_tensor(tensor_values, title, n_channel=0):
    if len(tensor_values.shape) > 3:
        tensor_values = tensor_values[:, :, :, n_channel]
        plt.imshow(tensor_values, cmap="gray")
        plt.title(title)
        plt.axis("off")
        plt.show()


def plot_images(image_channel):
    channels = list(image_channel.keys())
    for i in range(0, len(channels), 3):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        for j in range(3):
            channel_index = i + j
            if channel_index < len(channels):
                channel = channels[channel_index]
                images = image_channel[channel]
                try:
                    axs[j].imshow(images[0][0, :])
                    axs[j].set_title(f"Channel {channel}")
                    axs[j].axis("off")
                except Exception as e:
                    print(f"Error plotting channel {channel}: {e}")
                    axs[j].axis("off")
                    axs[j].text(0.5, 0.5, "Error", ha="center", va="center")
            else:
                axs[j].axis("off")
        plt.tight_layout()
        plt.show()


def plot_tensor_by_name(model, layer_name_list):

    with tf.Graph().as_default() as graph:
        # Import the model
        tf.import_graph_def(model.graph_def, name="")

        with tf.compat.v1.Session() as sess:
            for layer_name in layer_name_list:
                try:
                    # Get the tensor by name
                    tensor = sess.graph.get_tensor_by_name(f"{layer_name}:0")

                    # Get the tensor values
                    tensor_values = sess.run(tensor)

                    # Plot the tensor values
                    plot_tensor(tensor_values.squeeze(), layer_name)
                except KeyError:
                    # Handle the case where the tensor is not found
                    print(f"Tensor {layer_name} not found")


def codeocean_interactive_visualization(
    model,
    graph,
    sess,
    layer_name,
    channel,
    scope="",
    channels_first=False,
    thresholds=(512,),
    print_objectives=None,
    verbose=True,
):

    C = lambda neuron: objectives.channel(*neuron)

    tensor = graph.get_tensor_by_name(f"{layer_name}:0")
    tensor_shape = tensor.shape
    max_channel = tensor_shape[-1] - 1
    if 0 <= channel <= max_channel:
        clear_output(wait=True)
        objective_f = C((layer_name, channel))
        T = render.make_vis_T(
            model, objective_f, scope=scope, channels_first=channels_first
        )
        print_objective_func = render.make_print_objective_func(
            print_objectives, T
        )
        loss, vis_op, t_image = T("loss"), T("vis_op"), T("input")

        images = []

        img_ph = tf.compat.v1.placeholder(
            tf.float32, t_image.shape, "images_ph"
        )
        print("before sess", img_ph.shape)
        try:
            for i in range(max(thresholds) + 1):
                t_image_val = sess.run(t_image)
                print("going in ses", t_image_val.shape, type(t_image_val))
                loss_, _ = sess.run(
                    [loss, vis_op], feed_dict={img_ph: t_image_val}
                )
                if i in thresholds:
                    vis = t_image.eval()
                    images.append(vis)
                    if verbose:
                        print_objective_func(sess)
                        showing.show(np.hstack(vis))
        except KeyboardInterrupt:
            log.warn(f"Interrupted optimization at step {i+1:d}.")
            vis = t_image.eval()
            showing.show(np.hstack(vis))

        return images


def interactive_visualization(
    model,
    graph,
    layer_name,
    channel,
    scope="",
    channels_first=False,
):

    C = lambda neuron: objectives.channel(*neuron)

    tensor = graph.get_tensor_by_name(f"{layer_name}:0")
    tensor_shape = tensor.shape
    max_channel = tensor_shape[-1] - 1
    if 0 <= channel <= max_channel:
        clear_output(wait=True)
        _ = render.render_vis(
            model,
            C((layer_name, channel)),
            scope=scope,
            channels_first=channels_first,
        )


def batch_visualization(
    model,
    graph,
    layer_name,
    scope="",
    channels_first=False,
):
    C = lambda neuron: objectives.channel(*neuron)
    tensor = graph.get_tensor_by_name(f"{layer_name}:0")
    tensor_shape = tensor.shape
    # Check if the layer exists in the shape dictionary
    max_channel = tensor_shape[-1] - 1
    try:
        image_channel = {}
        for channel in tqdm(range(max_channel)):
            images = render.render_vis(
                model,
                C((layer_name, channel)),
                verbose=False,
                scope=scope,
                channels_first=channels_first,
            )
            image_channel[channel] = images
    except Exception:
        print("No gradients for this layer")
    return image_channel


def generate_gabor_kernels(frequencies, orientations):
    kernels = []
    for theta in orientations:
        for frequency in frequencies:
            kernel = gabor_kernel(frequency, theta=theta)
            kernels.append(kernel)
    return kernels


def apply_gabor_kernels(image, kernels):
    responses = []
    for kernel in kernels:
        filtered = convolve(image, np.real(kernel))
        responses.append(filtered)
    return responses


def extract_gabor_orientation(responses, orientations):
    orientation_map = np.zeros_like(responses[0])
    max_response = np.zeros_like(responses[0], dtype=np.float32)

    for i, response in enumerate(responses):
        orientation = orientations[i % len(orientations)]
        mask = response > max_response
        orientation_map[mask] = orientation
        max_response[mask] = response[mask]

    return orientation_map


def plot_gabor_kernels(frequencies, kernels):
    num_kernels = len(kernels)
    cols = len(frequencies)
    rows = len(kernels) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    for ax, kernel in zip(axes.flat, kernels):
        ax.imshow(np.real(kernel), cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    plt.show()
