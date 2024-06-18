import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import lucid_kietzmannlab.modelzoo.vision_models as models

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def _set_seeds(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_layer_channel_visualization(
    model: models.AlexNetCodeOcean, save_dir: str, random_seed: int = 1
):

    _set_seeds(random_seed)
    layer_shape_dict = model.layer_shape_dict
    save_dir = os.path.join(save_dir, "layer_channel_visualizations")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for index, layer_name in enumerate(tqdm(layer_shape_dict.keys())):
        model.vis_layer = layer_name
        image_channel = model.lucid_visualize_layer(batch=True)
        layer_dir = os.path.join(save_dir, f"layer_{index}")
        if not os.path.exists(layer_dir):
            os.makedirs(layer_dir)
        for channel, images in image_channel.items():
            image = images[0][0, :]
            clean_layer_name = layer_name.replace("/", "_")
            save_name = os.path.join(
                layer_dir, f"layer_{clean_layer_name}_channel_{channel}.png"
            )
            plt.imsave(save_name, image)
            plt.close()


if __name__ == "__main__":

    model_dir = "/Users/vkapoor/Downloads/models/AlexNet/training_seed_01"
    save_dir = "/Users/vkapoor/Downloads/models/AlexNet/"
    random_seed = 1
    model = models.AlexNetCodeOcean(
        model_dir=model_dir, random_seed=random_seed
    )
    save_layer_channel_visualization(model, save_dir, random_seed)
