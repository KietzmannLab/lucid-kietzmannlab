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


import itertools
import logging
import warnings
from collections import defaultdict
from collections.abc import Iterable

import numpy as np
import tensorflow as tf

from lucid_kietzmannlab.misc.io import loading, saving, showing
from lucid_kietzmannlab.modelzoo.util import (
    GraphDefHelper,
    extract_metadata,
    forget_xy,
    frozen_default_graph_def,
    infuse_metadata,
    load_graphdef,
)

log = logging.getLogger(__name__)


def recursive_enumerate_nd(it, stop_iter=None, prefix=()):
    """Recursively enumerate nested iterables with tuples n-dimenional indices.

    Args:
      it: object to be enumerated
      stop_iter: User defined funciton which can conditionally block further
        iteration. Defaults to allowing iteration.
      prefix: index prefix (not intended for end users)

    Yields:
      (tuple representing n-dimensional index, original iterator value)

    Example use:
      it = ((x+y for y in range(10) )
                 for x in range(10) )
      recursive_enumerate_nd(it) # yields things like ((9,9), 18)

    Example stop_iter:
      stop_iter = lambda x: isinstance(x, np.ndarray) and len(x.shape) <= 3
      # this prevents iteration into the last three levels (eg. x,y,channels) of
      # a numpy ndarray

    """
    if stop_iter is None:
        stop_iter = lambda x: False

    for n, x in enumerate(it):
        n_ = prefix + (n,)
        if isinstance(x, Iterable) and (not stop_iter(x)):
            yield from recursive_enumerate_nd(
                x, stop_iter=stop_iter, prefix=n_
            )
        else:
            yield (n_, x)


def dict_to_ndarray(d):
    """Convert a dictionary representation of an array (keys as indices) into a ndarray.

    Args:
      d: dict to be converted.

    Converts a dictionary representation of a sparse array into a ndarray. Array
    shape infered from maximum indices. Entries default to zero if unfilled.

    Example:
      >>> dict_to_ndarray({(0,0) : 3, (1,1) : 7})
      [[3, 0],
       [0, 7]]

    """
    assert len(d), "Dictionary passed to dict_to_ndarray() must not be empty."
    inds = list(d.keys())
    ind_dims = len(inds[0])
    assert all(len(ind) == ind_dims for ind in inds)
    ind_shape = [max(ind[i] + 1 for ind in inds) for i in range(ind_dims)]

    val0 = d[inds[0]]
    if isinstance(val0, np.ndarray):
        arr = np.zeros(ind_shape + list(val0.shape), dtype=val0.dtype)
    else:
        arr = np.zeros(ind_shape, dtype="float32")

    for ind, val in d.items():
        arr[ind] = val
    return arr


def batch_iter(it, batch_size=64):
    """Iterate through an iterable object in batches."""
    while True:
        batch = list(itertools.islice(it, batch_size))
        if not batch:
            break
        yield batch


def get_activations_iter(
    model,
    layer,
    generator,
    reducer="mean",
    batch_size=64,
    dtype=None,
    ind_shape=None,
    center_only=False,
):
    """Collect center activtions of a layer over many images from an iterable obj.

    Note: this is mostly intended for large synthetic families of images, where
      you can cheaply generate them in Python. For collecting activations over,
      say, ImageNet, there will be better workflows based on various dataset APIs
      in TensorFlow.

    Args:
      model: model for activations to be collected from.
      layer: layer (in model) for activtions to be collected from.
      generator: An iterable object (intended to be a generator) which produces
        tuples of the form (index, image). See details below.
      reducer: How to combine activations if multiple images map to the same index.
        Supports "mean", "rms", and "max".
      batch_size: How many images from the generator should be processes at once?
      dtype: determines dtype of returned data (defaults to model activation
        dtype). Can be used to make function memory efficient.
      ind_shape: Shape that indices can span. Optional, but makes function orders
        of magnitiude more memory efficient.

    Memory efficeincy:
      Using ind_shape is the main tool for make this function memory efficient.
      dtype="float16" can further help.

    Returns:
      A numpy array of shape [ind1, ind2, ..., layer_channels]
    """

    assert reducer in ["mean", "max", "rms"]
    combiner, normalizer = {
        "mean": (lambda a, b: a + b, lambda a, n: a / n),
        "rms": (lambda a, b: a + b**2, lambda a, n: np.sqrt(a / n)),
        "max": (lambda a, b: np.maximum(a, b), lambda a, n: a),
    }[reducer]

    with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
        t_img = tf.compat.v1.placeholder("float32", [None, None, None, 3])
        T = model.import_graph(t_img)
        t_layer = T(layer)

        responses = None
        count = None

        # # If we know the total length, let's give a progress bar
        # if ind_shape is not None:
        #   total = int(np.prod(ind_shape))
        #   generator = tqdm(generator, total=total)

        for batch in batch_iter(generator, batch_size=batch_size):

            inds, imgs = [x[0] for x in batch], [x[1] for x in batch]

            # Get activations (middle of image)
            acts = t_layer.eval({t_img: imgs})
            if center_only:
                acts = acts[:, acts.shape[1] // 2, acts.shape[2] // 2]
            if dtype is not None:
                acts = acts.astype(dtype)

            # On the first iteration of the loop, create objects to hold responses
            # (we wanted to know acts.shape[-1] before creating it in the numpy case)
            if responses is None:
                # If we don't know what the extent of the indices will be in advance
                # we need to use a dictionary to support dynamic range
                if ind_shape is None:
                    responses = {}
                    count = defaultdict(int)
                # But if we do, we can use a much more efficient numpy array
                else:
                    responses = np.zeros(
                        list(ind_shape) + list(acts.shape[1:]),
                        dtype=acts.dtype,
                    )
                    count = np.zeros(ind_shape, dtype=acts.dtype)

            # Store each batch item in appropriate index, performing reduction
            for ind, act in zip(inds, acts):
                count[ind] += 1
                if ind in responses:
                    responses[ind] = combiner(responses[ind], act)
                else:
                    responses[ind] = act

    # complete reduction as necessary, then return
    # First the case where everything is in numpy
    if isinstance(responses, np.ndarray):
        count = np.maximum(count, 1e-6)[..., None]
        return normalizer(responses, count)
    # Then the dynamic shape dictionary case
    else:
        for k in responses:
            count_ = np.maximum(count[k], 1e-6)[None].astype(acts.dtype)
            responses[k] = normalizer(responses[k], count_)
        return dict_to_ndarray(responses)


def _get_activations(
    model,
    layer,
    examples,
    batch_size=64,
    dtype=None,
    ind_shape=None,
    center_only=False,
):
    """Collect center activtions of a layer over an n-dimensional array of images.

    Note: this is mostly intended for large synthetic families of images, where
      you can cheaply generate them in Python. For collecting activations over,
      say, ImageNet, there will be better workflows based on various dataset APIs
      in TensorFlow.

    Args:
      model: model for activations to be collected from.
      layer: layer (in model) for activtions to be collected from.
      examples: A (potentially n-dimensional) array of images. Can be any nested
        iterable object, including a generator, as long as the inner most objects
        are a numpy array with at least 3 dimensions (image X, Y, channels=3).
      batch_size: How many images should be processed at once?
      dtype: determines dtype of returned data (defaults to model activation
        dtype). Can be used to make function memory efficient.
      ind_shape: Shape that the index (non-image) dimensions of examples. Makes
        code much more memory efficient if examples is not a numpy array.

    Memory efficeincy:
      Have examples be a generator rather than an array of images; this allows
      them to be lazily generated and not all stored in memory at once. Also
      use ind_shape so that activations can be stored in an efficient data
      structure. If you still have memory problems, dtype="float16" can probably
      get you another 2x.

    Returns:
      A numpy array of shape [ind1, ind2, ..., layer_channels]
    """

    if ind_shape is None and isinstance(examples, np.ndarray):
        ind_shape = examples.shape[:-3]

    # Create a generator which recursive enumerates examples, stoppping at
    # the third last dimesnion (ie. an individual iamge) if numpy arrays.
    examples_enumerated = recursive_enumerate_nd(
        examples,
        stop_iter=lambda x: isinstance(x, np.ndarray) and len(x.shape) <= 3,
    )

    # Get responses
    return get_activations_iter(
        model,
        layer,
        examples_enumerated,
        batch_size=batch_size,
        dtype=dtype,
        ind_shape=ind_shape,
        center_only=center_only,
    )


class Model:
    """Model allows using pre-trained models."""

    model_path = None
    labels_path = None
    checkpoint_path = None
    image_value_range = (-1, 1)
    image_shape = (None, None, 3)
    layers = ()
    model_name = None

    _labels = None
    _synset_ids = None
    _synsets = None
    _graph_def = None

    # Avoid pickling the in-memory graph_def.
    _blacklist = ["_graph_def"]

    def __getstate__(self):
        return {
            k: v for k, v in self.__dict__.items() if k not in self._blacklist
        }

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __eq__(self, other):
        if isinstance(other, Model):
            if self.checkpoint_path is None:
                return self.model_path == other.model_path
            if self.checkpoint_path is not None:
                return self.checkpoint_path == other.checkpoint_path
        return False

    def __hash__(self):
        if self.checkpoint_path is None:
            return hash(self.model_path)
        if self.checkpoint_path is not None:
            return hash(self.checkpoint_path)

    @property
    def labels(self):
        if not hasattr(self, "labels_path") or self.labels_path is None:
            raise RuntimeError(
                "This model does not have a labels_path specified!"
            )
        if not self._labels:
            self._labels = loading.load(self.labels_path, split=True)
        return self._labels

    @property
    def synset_ids(self):
        if not hasattr(self, "synsets_path") or self.synsets_path is None:
            raise RuntimeError(
                "This model does not have a synset_path specified!"
            )
        if not self._synset_ids:
            self._synset_ids = loading.load(self.synsets_path, split=True)
        return self._synset_ids

    @property
    def name(self):
        if self.model_name is None:
            return self.__class__.__name__
        else:
            return self.model_name

    def __str__(self):
        return self.name

    def to_json(self):
        return self.name  # TODO

    @property
    def graph_def(self):
        if not self._graph_def:
            self._graph_def = load_graphdef(self.model_path)
        return self._graph_def

    def load_graphdef(self):
        warnings.warn(
            "Calling `load_graphdef` is no longer necessary and now a noop. Graphs are loaded lazily when a models graph_def property is accessed.",
            DeprecationWarning,
        )

    def create_input(self, t_input=None, forget_xy_shape=True):
        """Create input tensor."""
        if t_input is None:
            t_input = tf.placeholder(tf.float32, self.image_shape)
        t_prep_input = t_input

        if len(t_prep_input.shape) == 3:
            t_prep_input = tf.expand_dims(t_prep_input, 0)
        if forget_xy_shape:
            t_prep_input = forget_xy(t_prep_input)
        if hasattr(self, "is_BGR") and self.is_BGR is True:
            t_prep_input = tf.reverse(t_prep_input, [-1])
        lo, hi = self.image_value_range
        t_prep_input = lo + t_prep_input * (hi - lo)
        return t_input, t_prep_input

    def import_graph(
        self,
        t_input=None,
        scope="import",
        forget_xy_shape=True,
        input_map=None,
    ):
        # print(scope)
        """Import model GraphDef into the current graph."""
        graph = tf.compat.v1.get_default_graph()
        assert graph.unique_name(scope, False) == scope, (
            'Scope "%s" already exists. Provide explicit scope names when '
            "importing multiple instances of the model."
        ) % scope

        t_input, t_prep_input = self.create_input(t_input, forget_xy_shape)

        final_input_map = {self.input_name: t_prep_input}
        if input_map is not None:
            final_input_map.update(input_map)

        tf.compat.v1.import_graph_def(
            self.graph_def, final_input_map, name=scope
        )

        def T(layer):
            if ":" in layer:
                return graph.get_tensor_by_name(f"{scope}/{layer}")
            else:
                return graph.get_tensor_by_name(f"{scope}/{layer}:0")

        return T

    def show_graph(self):
        if self.graph_def is None:
            raise Exception(
                "Model.show_graph(): Must load graph def before showing it."
            )
        showing.graph(self.graph_def)

    def get_layer(self, name):
        # Search by exact match
        for layer in self.layers:
            if layer.name == name:
                return layer
        # if not found by exact match, search fuzzy and warn user:
        for layer in self.layers:
            if name.lower() in layer.name.lower():
                log.warning(
                    "Found layer by fuzzy matching, please use '%s' in the future!",
                    layer.name,
                )
                return layer
        key_error_message = (
            "Could not find layer with name '{}'! Existing layer names are: {}"
        )
        layer_names = str([lay.name for lay in self.layers])
        raise KeyError(key_error_message.format(name, layer_names))

    @staticmethod
    def suggest_save_args(graph_def=None):
        if graph_def is None:
            graph_def = tf.get_default_graph().as_graph_def()
        gdhelper = GraphDefHelper(graph_def)
        inferred_info = dict.fromkeys(
            ("input_name", "image_shape", "output_names", "image_value_range")
        )
        node_shape = lambda n: [dim.size for dim in n.attr["shape"].shape.dim]
        potential_input_nodes = gdhelper.by_op["Placeholder"]
        output_nodes = [node.name for node in gdhelper.by_op["Softmax"]]

        if len(potential_input_nodes) == 1:
            input_node = potential_input_nodes[0]
            input_dtype = tf.dtypes.as_dtype(input_node.attr["dtype"].type)
            if input_dtype.is_floating:
                input_name = input_node.name
                print(
                    f"Inferred: input_name = {input_name} (because it was the only Placeholder in the graph_def)"
                )
                inferred_info["input_name"] = input_name
            else:
                print(
                    "Warning: found a single Placeholder, but its dtype is {}. Lucid's parameterizations can only replace float dtypes. We're now scanning to see if you maybe divide this placeholder by 255 to get a float later in the graph...".format(
                        str(input_node.attr["dtype"]).strip()
                    )
                )
                neighborhood = gdhelper.neighborhood(input_node, degree=5)
                divs = [n for n in neighborhood if n.op == "RealDiv"]
                consts = [n for n in neighborhood if n.op == "Const"]
                magic_number_present = any(
                    255 in c.attr["value"].tensor.int_val for c in consts
                )
                if divs and magic_number_present:
                    if len(divs) == 1:
                        input_name = divs[0].name
                        print(
                            f"Guessed: input_name = {input_name} (because it's the only division by 255 near the only placeholder)"
                        )
                        inferred_info["input_name"] = input_name
                        image_value_range = (0, 1)
                        print(
                            f"Guessed: image_value_range = {image_value_range} (because you're dividing by 255 near the only placeholder)"
                        )
                        inferred_info["image_value_range"] = (0, 1)
                    else:
                        warnings.warn(
                            f"Could not infer input_name because there were multiple division ops near your the only placeholder. Candidates include: {[n.name for n in divs]}"
                        )
        else:
            warnings.warn(
                "Could not infer input_name because there were multiple or no Placeholders."
            )

        if inferred_info["input_name"] is not None:
            input_node = gdhelper.by_name[inferred_info["input_name"]]
            shape = node_shape(input_node)
            if len(shape) in (3, 4):
                if len(shape) == 4:
                    shape = shape[1:]
                if -1 not in shape:
                    print(f"Inferred: image_shape = {shape}")
                    inferred_info["image_shape"] = shape
            if inferred_info["image_shape"] is None:
                warnings.warn("Could not infer image_shape.")

        if output_nodes:
            print(
                f"Inferred: output_names = {output_nodes}  (because those are all the Softmax ops)"
            )
            inferred_info["output_names"] = output_nodes
        else:
            warnings.warn("Could not infer output_names.")

        report = []
        report.append(
            "# Please sanity check all inferred values before using this code."
        )
        report.append(
            "# Incorrect `image_value_range` is the most common cause of feature visualization bugs! Most methods will fail silently with incorrect visualizations!"
        )
        report.append("Model.save(")

        suggestions = {
            "input_name": "input",
            "image_shape": [224, 224, 3],
            "output_names": ["logits"],
            "image_value_range": "[-1, 1], [0, 1], [0, 255], or [-117, 138]",
        }
        for key, value in inferred_info.items():
            if value is not None:
                report.append(f"    {key}={value!r},")
            else:
                report.append(
                    f"    {key}=_,                   # TODO (eg. {suggestions[key]!r})"
                )
        report.append("  )")

        print("\n".join(report))
        return inferred_info

    @staticmethod
    def save(
        save_url, input_name, output_names, image_shape, image_value_range
    ):
        if ":" in input_name:
            raise ValueError(
                "input_name appears to be a tensor (name contains ':') but must be an op."
            )
        if any([":" in name for name in output_names]):
            raise ValueError(
                "output_namess appears to be contain tensor (name contains ':') but must be ops."
            )

        metadata = {
            "input_name": input_name,
            "image_shape": image_shape,
            "image_value_range": image_value_range,
        }

        graph_def = frozen_default_graph_def([input_name], output_names)
        infuse_metadata(graph_def, metadata)
        saving.save(graph_def, save_url)

    @staticmethod
    def load(url):
        if url.endswith(".pb"):
            return Model.load_from_graphdef(url)
        elif url.endswith(".json"):
            return Model.load_from_manifest(url)

    @staticmethod
    def load_from_metadata(model_url, metadata):
        class DynamicModel(Model):
            model_path = model_url
            input_name = metadata["input_name"]
            image_shape = metadata["image_shape"]
            image_value_range = metadata["image_value_range"]

        return DynamicModel()

    @staticmethod
    def load_from_graphdef(graphdef_url):
        graph_def = loading.load(graphdef_url)
        metadata = extract_metadata(graph_def)
        if metadata:
            return Model.load_from_metadata(graphdef_url, metadata)
        else:
            raise ValueError(
                f"Model.load_from_graphdef was called on a GraphDef ({graphdef_url}) that does not contain Lucid's metadata node. Model.load only works for models saved via Model.save. For the graphdef you're trying to load, you will need to provide custom metadata; see Model.load_from_metadata()"
            )

    def get_activations(
        self,
        layer,
        examples,
        batch_size=64,
        dtype=None,
        ind_shape=None,
        center_only=False,
    ):
        """Collect center activtions of a layer over an n-dimensional array of images.

        Note: this is mostly intended for large synthetic families of images, where
          you can cheaply generate them in Python. For collecting activations over,
          say, ImageNet, there will be better workflows based on various dataset APIs
          in TensorFlow.

        Args:
          layer: layer (in model) for activtions to be collected from.
          examples: A (potentially n-dimensional) array of images. Can be any nested
            iterable object, including a generator, as long as the inner most objects
            are a numpy array with at least 3 dimensions (image X, Y, channels=3).
          batch_size: How many images should be processed at once?
          dtype: determines dtype of returned data (defaults to model activation
            dtype). Can be used to make function memory efficient.
          ind_shape: Shape that the index (non-image) dimensions of examples. Makes
            code much more memory efficient if examples is not a numpy array.

        Memory efficeincy:
          Have examples be a generator rather than an array of images; this allows
          them to be lazily generated and not all stored in memory at once. Also
          use ind_shape so that activations can be stored in an efficient data
          structure. If you still have memory problems, dtype="float16" can probably
          get you another 2x.

        Returns:
          A numpy array of shape [ind1, ind2, ..., layer_channels]
        """

        return _get_activations(
            self,
            layer,
            examples,
            batch_size=batch_size,
            dtype=dtype,
            ind_shape=ind_shape,
            center_only=center_only,
        )
