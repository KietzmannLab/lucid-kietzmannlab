import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
from scipy.misc import face
from skimage.transform import resize

import lucid_kietzmannlab.modelzoo.vision_models as models


def model_tensor_plot(
    sess,
    model,
    tensors_to_plot,
    input_placeholder,
    input_data,
    channels_first=False,
    feed_dict={},
):
    # Add input data to the feed dictionary
    feed_dict[input_placeholder] = input_data

    # Run the model to get the values of the specified tensors
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

            # Initialize variables
            sess.run(tf.compat.v1.global_variables_initializer())

            # Assuming input placeholder name is "Placeholder"
            input_placeholder = model.graph.get_tensor_by_name("Placeholder:0")

            # Find and set the placeholder for 'load_data/Placeholder'
            boolean_placeholder = model.graph.get_tensor_by_name(
                "load_data/Placeholder:0"
            )

            # Create feed dictionary for placeholders
            feed_dict = {
                boolean_placeholder: False  # Adjust as needed for your use case
            }

            model_tensor_plot(
                sess,
                model,
                tensors_to_plot,
                input_placeholder,
                input_data,
                channels_first=channels_first,
                feed_dict=feed_dict,
            )


tf.compat.v1.disable_eager_execution()


model_checkpoint_dir = "/share/klab/vkapoor/AlexNet/training_seed_05"
training_seed = 5
model = models.AlexNetv2(
    model_checkpoint_dir=model_checkpoint_dir, training_seed=training_seed
)

input_shape = model.image_shape
face_image = face()
face_image = np.transpose(face_image, (2, 0, 1))
resized_face_image = np.zeros((3, 227, 227))
for i in range(3):
    resized_face_image[i] = np.array(resize(face_image[i], (227, 227)))

plot_selected_layer_tensors(
    model,
    np.expand_dims(resized_face_image, axis=0),
    tensors_to_plot=["tower_0/alexnet_v2/conv4/Conv2D"],
    channels_first=True,
)
