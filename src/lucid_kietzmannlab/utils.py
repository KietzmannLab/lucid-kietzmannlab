import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf


def plot_tensor(tensor_values, title, n_channel=0):
    if len(tensor_values.shape) > 3:
        tensor_values = tensor_values[:, :, :, n_channel]
        plt.imshow(tensor_values, cmap="gray")
        plt.title(title)
        plt.axis("off")
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
