from timeit import default_timer as timer

import numpy as np
import tensorflow as tf
from scipy import ndimage


def resize(image, target_size=None, ratios=None, **kwargs):
    """Resize an ndarray image of rank 3 or 4.
    target_size can be a tuple `(width, height)` or scalar `width`.
    Alternatively you can directly specify the ratios by which each
    dimension should be scaled, or a single ratio"""

    # input validation
    if target_size is None:
        assert ratios is not None
    else:
        if isinstance(target_size, int):
            target_size = (target_size, target_size)

        if not isinstance(target_size, (list, tuple, np.ndarray)):
            message = (
                "`target_size` should be a single number (width) or a list"
                "/tuple/ndarray (width, height), not {}.".format(
                    type(target_size)
                )
            )
            raise ValueError(message)

    rank = len(image.shape)
    assert 3 <= rank <= 4

    original_size = image.shape[-3:-1]

    ratios_are_noop = (
        all(ratio == 1 for ratio in ratios) if ratios is not None else False
    )
    target_size_is_noop = (
        target_size == original_size if target_size is not None else False
    )
    if ratios_are_noop or target_size_is_noop:
        return image  # noop return because ndimage.zoom doesn't check itself

    # TODO: maybe allow -1 in target_size to signify aspect-ratio preserving resize?
    ratios = ratios or [t / o for t, o in zip(target_size, original_size)]
    zoom = [1] * rank
    zoom[-3:-1] = ratios

    roughly_resized = ndimage.zoom(image, zoom, **kwargs)
    if target_size is not None:
        return roughly_resized[..., : target_size[0], : target_size[1], :]
    else:
        return roughly_resized


def image_activations(model, image, layer_names=None, scope="import"):
    if layer_names is None:
        layer_names = [
            layer["name"] for layer in model.layers if "conv" in layer.tags
        ]

    resized_image = resize(image, model.image_shape[:2])
    print(resized_image.shape)
    with tf.Graph().as_default() as graph, tf.compat.v1.Session() as sess:
        image_t = tf.compat.v1.placeholder_with_default(
            resized_image, shape=resized_image.shape
        )
        model.import_graph(image_t, scope="import")
        layer_ts = {}
        for layer_name in layer_names:
            name = (
                layer_name if layer_name.endswith(":0") else layer_name + ":0"
            )
            layer_t = graph.get_tensor_by_name(f"{scope}" / {}.format(name))[0]
            layer_ts[layer_name] = layer_t
        activations = sess.run(layer_ts)

    return activations


def manifest_image_activations(model, image, **kwargs):
    start = timer()
    activations_dict = image_activations(model, image, **kwargs)
    end = timer()
    elapsed = end - start

    results = {"type": "image-activations", "took": elapsed}

    results["values"] = [
        {
            "type": "activations",
            "value": value,
            "shape": value.shape,
            "layer_name": layer_name,
        }
        for layer_name, value in activations_dict.items()
    ]

    return results
