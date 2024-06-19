import os
import random
from concurrent.futures import ThreadPoolExecutor

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


def save_image(save_info):
    layer_dir, clean_layer_name, channel, image = save_info
    save_name = os.path.join(
        layer_dir, f"layer_{clean_layer_name}_channel_{channel}.png"
    )
    plt.imsave(save_name, image)
    plt.close()


def save_layer_channel_visualization(
    model: models.AlexNetCodeOcean,
    save_dir: str,
    random_seed: int = 1,
    channel_start: int = 0,
    channel_end: int = -1,
    max_workers: int = 4,
):
    _set_seeds(random_seed)
    layer_shape_dict = model.layer_shape_dict
    save_dir = os.path.join(save_dir, "layer_channel_visualizations")
    os.makedirs(save_dir, exist_ok=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, layer_name in enumerate(tqdm(layer_shape_dict.keys())):
            if index > 0:
                model.vis_layer = layer_name
                image_channel = model.lucid_visualize_layer(
                    batch=True,
                    channel_start=channel_start,
                    channel_end=channel_end,
                )
                layer_dir = os.path.join(save_dir, f"layer_{index + 1}")
                os.makedirs(layer_dir, exist_ok=True)

                clean_layer_name = layer_name.replace("/", "_")

                save_infos = []
                for channel, images in image_channel.items():
                    if len(images) > 0:
                        image = images[0][0, :]
                        save_infos.append(
                            (layer_dir, clean_layer_name, channel, image)
                        )
                # Save images for the current layer
                list(
                    tqdm(
                        executor.map(save_image, save_infos),
                        total=len(save_infos),
                        desc=f"Saving {layer_name}",
                    )
                )


if __name__ == "__main__":
    model_dir = (
        "/Users/vkapoor/Downloads/hpc/vkapoor/AlexNet/training_seed_01/"
    )
    save_dir = "/Users/vkapoor/Downloads/hpc/vkapoor/AlexNet/"
    random_seed = 1
    channel_start = 0
    channel_end = -1
    max_workers = 8
    model = models.AlexNetCodeOcean(
        model_dir=model_dir, random_seed=random_seed
    )
    save_layer_channel_visualization(
        model, save_dir, random_seed, channel_start, channel_end, max_workers
    )
