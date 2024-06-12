import numpy as np
import tensorflow.compat.v1 as tf
from scipy.misc import face
from skimage.transform import resize

import lucid_kietzmannlab.modelzoo.vision_models as models
from lucid_kietzmannlab.modelzoo.vision_models import (
    plot_selected_layer_tensors,
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
