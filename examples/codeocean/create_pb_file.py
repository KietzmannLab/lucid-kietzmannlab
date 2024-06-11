import os
from pathlib import Path

import tensorflow as tf


def create_pb_model_file(
    meta_path, model_checkpoint_dir, save_path, save_name
):

    with tf.compat.v1.Session() as sess:
        saver = tf.compat.v1.train.import_meta_graph(
            meta_path, clear_devices=True
        )
        saver.restore(sess, model_checkpoint_dir)

        output_node_names = [
            n.name
            for n in tf.compat.v1.get_default_graph().as_graph_def().node
        ]
        frozen_graph_def = (
            tf.compat.v1.graph_util.convert_variables_to_constants(
                sess, sess.graph_def, output_node_names
            )
        )

        with open(os.path.join(save_path, save_name + ".pb"), "wb") as f:
            f.write(frozen_graph_def.SerializeToString())


if __name__ == "__main__":

    parent_dir = Path(__file__).parent
    training_seed = 5
    model_checkpoint_dir = os.path.join(
        parent_dir, "models/AlexNet/training_seed_05"
    )
    meta_path = os.path.join(model_checkpoint_dir, "model.ckpt_epoch89.meta")
    save_path = os.path.join(parent_dir, "models/AlexNet/")
    model_checkpoint_file = os.path.join(
        model_checkpoint_dir, "model.ckpt_epoch89"
    )
    save_name = "seed5"
    create_pb_model_file(
        meta_path, model_checkpoint_file, save_path, save_name
    )
