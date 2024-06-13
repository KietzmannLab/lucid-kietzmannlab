import os
from subprocess import call

DNN_architecture = "AlexNet"
training_set = "ecoset"
training_seed = 1
layers = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
]  # Alexnet layers 1-7, [8:9] AlexNet output before (8) and after (9) softmax

image_dir_parent = "/Users/vkapoor/Downloads/input_image_sets/"
IMAGE_DIRS = [
    "cichy_92_only_4_stimuli_debugging",
    "cichy_92",
    "horikawa_1200_training",
]
IMAGE_DIR = os.path.join(
    image_dir_parent, IMAGE_DIRS[1]
)  # "0" - 4 test images, "1" - 92 stimuli from Cichy et al. 2014, "2" - 1200 stimuli from Horikawa 2018
MODEL_DIR = (
    f"/Users/vkapoor/Downloads/models/AlexNet/training_seed_0{training_seed}/"
)
DISTANCE_MEASURE = "correlation"  # in the ecoset paper we used correlation distance ("correlation") to compute RDMs. Other commonly used options include (more options available via scipy.spatial.distance.pdist): ‘cityblock’, ‘correlation’, ‘cosine’, ‘euclidean’
compute_RDMs_bool = 1


for layer in layers:
    output_size = 565
    CKPT_PATH = os.path.join(MODEL_DIR, "model.ckpt_epoch89")

    call(
        [
            "python",
            "extract_activations.py",
            CKPT_PATH,
            str(output_size),
            str(training_seed),
            DISTANCE_MEASURE,
            IMAGE_DIR,
            training_set,
            str(layer - 1),
            str(compute_RDMs_bool),
        ]
    )
