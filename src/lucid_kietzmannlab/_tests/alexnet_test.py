import tensorflow.compat.v1 as tf
import tf_slim as slim

from lucid_kietzmannlab.modelzoo.vision_models import alexnet_v2

tf.compat.v1.disable_eager_execution()


class AlexnetV2Test(tf.test.TestCase):

    def testBuild(self):
        batch_size = 5
        height, width = 224, 224
        num_classes = 1000
        with self.test_session():
            inputs = tf.random.uniform((batch_size, height, width, 3))
            logits, _ = alexnet_v2(inputs, num_classes)
            self.assertListEqual(
                logits.get_shape().as_list(), [batch_size, num_classes]
            )

    def testFullyConvolutional(self):
        batch_size = 1
        height, width = 300, 400
        num_classes = 1000
        with self.test_session():
            inputs = tf.random.uniform((batch_size, height, width, 3))
            logits, _ = alexnet_v2(inputs, num_classes, spatial_squeeze=False)
            self.assertListEqual(
                logits.get_shape().as_list(), [batch_size, 4, 7, num_classes]
            )

    def testGlobalPool(self):
        batch_size = 1
        height, width = 256, 256
        num_classes = 1000
        with self.test_session():
            inputs = tf.random.uniform((batch_size, height, width, 3))
            logits, _ = alexnet_v2(
                inputs, num_classes, spatial_squeeze=False, global_pool=True
            )
            self.assertListEqual(
                logits.get_shape().as_list(), [batch_size, 1, 1, num_classes]
            )

    def testEndPoints(self):
        batch_size = 5
        height, width = 224, 224
        num_classes = 1000
        with self.test_session():
            inputs = tf.random.uniform((batch_size, height, width, 3))
            _, end_points = alexnet_v2(inputs, num_classes)
            expected_names = [
                "alexnet_v2/conv1",
                "alexnet_v2/pool1",
                "alexnet_v2/conv2",
                "alexnet_v2/pool2",
                "alexnet_v2/conv3",
                "alexnet_v2/conv4",
                "alexnet_v2/conv5",
                "alexnet_v2/pool5",
                "alexnet_v2/fc6",
                "alexnet_v2/fc7",
                "alexnet_v2/fc8",
            ]
            self.assertSetEqual(set(end_points.keys()), set(expected_names))

    def testNoClasses(self):
        batch_size = 5
        height, width = 224, 224
        num_classes = None
        with self.test_session():
            inputs = tf.random.uniform((batch_size, height, width, 3))
            net, end_points = alexnet_v2(inputs, num_classes)
            expected_names = [
                "alexnet_v2/conv1",
                "alexnet_v2/pool1",
                "alexnet_v2/conv2",
                "alexnet_v2/pool2",
                "alexnet_v2/conv3",
                "alexnet_v2/conv4",
                "alexnet_v2/conv5",
                "alexnet_v2/pool5",
                "alexnet_v2/fc6",
                "alexnet_v2/fc7",
                "alexnet_v2/fc8",
            ]
            self.assertSetEqual(set(end_points.keys()), set(expected_names))
            self.assertTrue(net.op.name.startswith("alexnet_v2/fc8"))
            self.assertListEqual(net.get_shape().as_list(), [batch_size, 4096])

    def testModelVariables(self):
        batch_size = 5
        height, width = 224, 224
        num_classes = 1000
        with self.test_session():
            inputs = tf.random.uniform((batch_size, height, width, 3))
            alexnet_v2(inputs, num_classes)
            expected_names = [
                "alexnet_v2/conv1/weights",
                "alexnet_v2/conv1/biases",
                "alexnet_v2/conv2/weights",
                "alexnet_v2/conv2/biases",
                "alexnet_v2/conv3/weights",
                "alexnet_v2/conv3/biases",
                "alexnet_v2/conv4/weights",
                "alexnet_v2/conv4/biases",
                "alexnet_v2/conv5/weights",
                "alexnet_v2/conv5/biases",
                "alexnet_v2/fc6/weights",
                "alexnet_v2/fc6/biases",
                "alexnet_v2/fc7/weights",
                "alexnet_v2/fc7/biases",
                "alexnet_v2/fc8/weights",
                "alexnet_v2/fc8/biases",
            ]
            model_variables = [v.op.name for v in slim.get_model_variables()]
            self.assertSetEqual(set(model_variables), set(expected_names))

    def testEvaluation(self):
        batch_size = 2
        height, width = 224, 224
        num_classes = 1000
        with self.test_session():
            eval_inputs = tf.random.uniform((batch_size, height, width, 3))
            logits, _ = alexnet_v2(eval_inputs, is_training=False)
            self.assertListEqual(
                logits.get_shape().as_list(), [batch_size, num_classes]
            )
            predictions = tf.argmax(input=logits, axis=1)
            self.assertListEqual(
                predictions.get_shape().as_list(), [batch_size]
            )

    def testTrainEvalWithReuse(self):
        train_batch_size = 2
        eval_batch_size = 1
        train_height, train_width = 224, 224
        eval_height, eval_width = 300, 400
        num_classes = 1000
        with self.test_session():
            train_inputs = tf.random.uniform(
                (train_batch_size, train_height, train_width, 3)
            )
            logits, _ = alexnet_v2(train_inputs)
            self.assertListEqual(
                logits.get_shape().as_list(), [train_batch_size, num_classes]
            )
            tf.get_variable_scope().reuse_variables()
            eval_inputs = tf.random.uniform(
                (eval_batch_size, eval_height, eval_width, 3)
            )
            logits, _ = alexnet_v2(
                eval_inputs, is_training=False, spatial_squeeze=False
            )
            self.assertListEqual(
                logits.get_shape().as_list(),
                [eval_batch_size, 4, 7, num_classes],
            )
            logits = tf.reduce_mean(input_tensor=logits, axis=[1, 2])
            predictions = tf.argmax(input=logits, axis=1)
            print(predictions.shape)

    def testForward(self):
        batch_size = 1
        height, width = 224, 224
        with self.test_session() as sess:
            inputs = tf.random.uniform((batch_size, height, width, 3))
            logits, _ = alexnet_v2(inputs)
            sess.run(tf.global_variables_initializer())
            output = sess.run(logits)
            self.assertTrue(output.any())


if __name__ == "__main__":
    # alex = AlexnetV2Test()
    # alex.testBuild()
    tf.test.main()
