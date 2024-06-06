# %%
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from IPython.display import clear_output
from ipywidgets import Dropdown, IntSlider, interact
from tqdm import tqdm

import lucid_kietzmannlab.modelzoo.vision_models as models
import lucid_kietzmannlab.optvis.objectives as objectives
import lucid_kietzmannlab.optvis.render as render

# %% [markdown]
# ## A notebook to maximally activate certain neurons in the Alex Net neural network

# %% [markdown]
# ### Choose the alexnet model

# %%
model_checkpoint_dir = "/share/klab/vkapoor/training_seed_05"
model_checkpoint = "model.ckpt_epoch89"

# %%
model = models.AlexNet(model_checkpoint_dir, model_checkpoint)
model.load_graphdef()

# %%
layer_name_list = [
    node.name for node in model.graph_def.node if "input" not in node.name
]

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


C = lambda neuron: objectives.channel(*neuron)


def visualize(layer_name, channel):
    # Check if the layer exists in the shape dictionary
    if layer_name in layer_shape_dict:
        # Check if the selected channel is within bounds
        print(layer_shape_dict[layer_name])
        max_channel = layer_shape_dict[layer_name][-1] - 1
        if 0 <= channel <= max_channel:
            clear_output(wait=True)
            # Render visualization for the selected layer and channel
            try:
                _ = render.render_vis(model, C((layer_name, channel)))
            except Exception:
                print("No gradients for this layer")


def visualize_all():
    # Check if the layer exists in the shape dictionary
    layer_name = current_dropdown_value({"new": layer_dropdown.value})
    print(layer_name)
    if layer_name in layer_shape_dict:
        # Check if the selected channel is within bounds
        try:
            image_channel = {}
            for channel in tqdm(range(channel_slider.max)):
                images = render.render_vis(
                    model, C((layer_name, channel)), verbose=False
                )
                image_channel[channel] = images
        except Exception:
            print("No gradients for this layer")
    return image_channel


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


# Create dropdown menu for layer selection
layer_dropdown = Dropdown(
    options=list(layer_shape_dict.keys()), description="Layer:"
)

# Create slider for channel selection
channel_slider = IntSlider(min=0, max=0, description="Channel:")


def update_channel_slider(change):
    layer_name = change.new
    if layer_name in layer_shape_dict:

        max_channel = layer_shape_dict[layer_name][-1] - 1
        channel_slider.max = max_channel


def current_slider_value(*args):
    return channel_slider.value


def current_dropdown_value(change):
    return change["new"]


channel_slider.observe(current_slider_value, names="value")
layer_dropdown.observe(current_dropdown_value, names="value")


# %% [markdown]
# ### In the code block below a user can interactively choose the layer to maximally activate the neuron of and then visualize it for a certain channel

# %%
layer_dropdown.observe(update_channel_slider, names="value")

# Create an interactive visualization
interact(visualize, layer_name=layer_dropdown, channel=channel_slider)


# %% [markdown]
# In the code below we make a non-interactive plot of all the channels at that layer at once (Memory intensive)

# %%
image_channel = visualize_all()


# %% [markdown]
# ### Visualize neuron activations for all the channels in the layer

# %%

if image_channel:
    plot_images(image_channel)

# %%
