# lucid-kietzmannlab

[![License BSD-3](https://img.shields.io/pypi/l/lucid-kietzmannlab.svg?color=green)](https://github.com/KietzmannLab/lucid-kietzmannlab/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/lucid-kietzmannlab.svg?color=green)](https://pypi.org/project/lucid-kietzmannlab)
[![Python Version](https://img.shields.io/pypi/pyversions/lucid-kietzmannlab.svg?color=green)](https://python.org)
[![codecov](https://codecov.io/gh/KietzmannLab/lucid-kietzmannlab/branch/main/graph/badge.svg)](https://codecov.io/gh/KietzmannLab/lucid-kietzmannlab)


Visualization of Interaction Between Neurons using Lucid

----------------------------------

This [KietzmannLab] package was generated with [Cookiecutter] using [@KietzmannLab]'s [cookiecutter-template] template.

## Algorithm

Starting from random noise, we optimize an image to activate a particular neuron. A neuron is a certain layer of the trained network at a certain channel. The optimization function used is the negative of the spatial activation map at that layer and channel and after iterating for 512 steps a random noise image is transformed to the features that maximally activate that particular neuron.

Using this approach we can convincingly show that earlier neural network layers maximally activate Gabor like features (edges at different orientation and scale) while the later layers are more feature rich containing object level features.

By visualizing different channles of a certain layer we show the diversity captured by the trained model. In this repository we use pre-trained Alexnet models [trained on ecoset](https://codeocean.com/capsule/9570390/tree/v1). Using any pre-trained model seeds the maximal activations of different layers of AlexNet model can be seen [interactively](examples/interactive_layer_visualization.ipynb) or by iterating over all the layers and saving the results as png files for each layer and channel using this [script](examples/ecoset_layers_activation.py). Prior to running the notebook or the script please install the package in your virtual environment by following the instructions below.

## Installation

You can install `lucid-kietzmannlab` via [pip]:

    pip install git+https://github.com/KietzmannLab/lucid-kietzmannlab.git
    
It is recommended to do the installation in a virtual conda environment, if you do not have a conda environment you can create one using:

            conda create -n conda_env_name python=3.12
            
 Then activate the environment and install the package using pip as mentioned above.



## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"lucid-kietzmannlab" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.


[pip]: https://pypi.org/project/pip/
[KietzmannLab]: https://github.com/KietzmannLab/
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@KietzmannLab]: https://github.com/KietzmannLab/
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-template]: https://github.com/KietzmannLab/cookiecutter-kietzmannlab-template

[file an issue]: https://github.com/KietzmannLab/lucid-kietzmannlab/issues

[KietzmannLab]: https://github.com/KietzmannLab/
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
