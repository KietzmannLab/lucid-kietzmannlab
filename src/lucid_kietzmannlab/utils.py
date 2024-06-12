import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from IPython.display import clear_output
from tqdm import tqdm

import lucid_kietzmannlab.optvis.objectives as objectives
import lucid_kietzmannlab.optvis.render as render


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


def interactive_visualization(
    model,
    layer_name,
    channel,
    scope="",
    channels_first=False,
):

    C = lambda neuron: objectives.channel(*neuron)
    graph = tf.compat.v1.get_default_graph()
    tensor = graph.get_tensor_by_name(f"{layer_name}:0")
    tensor_shape = tensor.shape
    # Check if the layer exists in the shape dictionary
    max_channel = tensor_shape[-1] - 1
    if 0 <= channel <= max_channel:
        clear_output(wait=True)
        # Render visualization for the selected layer and channel
        _ = render.render_vis(
            model,
            C((layer_name, channel)),
            scope=scope,
            channels_first=channels_first,
        )


def batch_visualization(
    model,
    layer_name,
    layer_shape_dict,
    channel_slider,
    scope="",
    channels_first=False,
):
    C = lambda neuron: objectives.channel(*neuron)
    if layer_name in layer_shape_dict:
        # Check if the selected channel is within bounds
        try:
            image_channel = {}
            for channel in tqdm(range(channel_slider.max)):
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
