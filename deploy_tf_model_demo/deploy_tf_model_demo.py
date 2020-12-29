"""
TODO: docstring
"""
import grpc
import os
import sys
import tensorflow

class Client:
    """
    Send JPEG image to tensorflow_model_server loaded with inception model.
    """
    def __init__(self):
        """
        TODO: docstring
        """
        tensorflow.app.flags.DEFINE_string('server', 'localhost:9000', 'PredictionService host:port')
        tensorflow.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
        FLAGS = tensorflow.app.flags.FLAGS

    def __call__(self):
        """
        TODO: docstring
        """
        host, port = FLAGS.server.split(':')
        channel = grpc.beta.implementations.insecure_channel(host, int(port))
        stub = tensorflow_serving.apis.prediction_service_pb2.beta_create_PredictionService_stub(channel)
        with open(FLAGS.image, 'rb') as f:
            data = f.read()
            request = tensorflow_serving.apis.predict_pb2.PredictRequest()
            request.model_spec.name = 'inception'
            request.model_spec.signature_name = 'predict_images'
            request.inputs['images'].CopyFrom(
                tensorflow.contrib.util.make_tensor_proto(data, shape=[1]))
            result = stub.Predict(request, 10.0)
            print(result)

class CustomModel:
    """
    Train and export a simple Softmax Regression TensorFlow model.
    The model is from the TensorFlow "MNIST For ML Beginner" tutorial. This program
    simply follows all its training instructions, and uses TensorFlow SavedModel to
    export the trained model with proper signatures that can be loaded by standard
    tensorflow_model_server.
    Usage: mnist_export.py [--training_iteration=x] [--model_version=y] export_dir
    """
    def __init__(self):
        """
        TODO: docstring
        """
        tensorflow.app.flags.DEFINE_integer('training_iteration', 1000, 'number of training iterations.')
        tensorflow.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
        tensorflow.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory.')
        FLAGS = tensorflow.app.flags.FLAGS

    def __call__(self):
        """
        TODO: docstring
        """
        if len(sys.argv) < 2 or sys.argv[-1].startswith('-'):
            print('Usage: mnist_export.py [--training_iteration=x] [--model_version=y] export_dir')
            sys.exit(-1)
        if FLAGS.training_iteration <= 0:
            print('Please specify a positive value for training iteration.')
            sys.exit(-1)
        if FLAGS.model_version <= 0:
            print('Please specify a positive value for version number.')
            sys.exit(-1)
        # train model
        print('Training model ...')
        mnist = tensorflow_serving.example.mnist_input_data.read_data_sets(FLAGS.work_dir, one_hot=True)
        sess = tensorflow.InteractiveSession()
        serialized_tf_example = tensorflow.placeholder(tensorflow.string, name='tf_example')
        feature_configs = {'x': tensorflow.FixedLenFeature(shape=[784], dtype=tensorflow.float32),}
        tf_example = tensorflow.parse_example(serialized_tf_example, feature_configs)
        # build model
        x = tensorflow.identity(tf_example['x'], name='x')
        y_ = tensorflow.placeholder('float', shape=[None, 10])
        w = tensorflow.Variable(tensorflow.zeros([784, 10]))
        b = tensorflow.Variable(tensorflow.zeros([10]))
        sess.run(tensorflow.global_variables_initializer())
        y = tensorflow.nn.softmax(tensorflow.matmul(x, w) + b, name='y')
        cross_entropy = -tensorflow.reduce_sum(y_ * tensorflow.log(y))
        train_step = tensorflow.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
        values, indices = tensorflow.nn.top_k(y, 10)
        table = tensorflow.contrib.lookup.index_to_string_table_from_tensor(
            tensorflow.constant([str(i) for i in xrange(10)]))
        # train model
        prediction_classes = table.lookup(tensorflow.to_int64(indices))
        for _ in range(FLAGS.training_iteration):
            batch = mnist.train.next_batch(50)
            train_step.run(feed_dict={x: batch[0], y_: batch[1]})
        correct_prediction = tensorflow.equal(tensorflow.argmax(y, 1), tensorflow.argmax(y_, 1))
        accuracy = tensorflow.reduce_mean(tensorflow.cast(correct_prediction, 'float'))
        print('training accuracy %g' % sess.run(
            accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
        print('Done training.')
        # save model
        export_path_base = sys.argv[-1]
        export_path = os.path.join(
            tensorflow.python.util.compat.as_bytes(export_path_base),
            tensorflow.python.util.compat.as_bytes(str(FLAGS.model_version)))
        print('Exporting trained model to', export_path)
        # this creates a SERVABLE from our model
        # saves a "snapshot" of the trained model to reliable storage 
        builder = tensorflow.python.saved_model.builder.SavedModelBuilder(export_path)
        # build the signature_def_map
        classification_inputs = tensorflow.python.saved_model.utils.build_tensor_info(serialized_tf_example)
        classification_outputs_classes = tensorflow.python.saved_model.utils.build_tensor_info(prediction_classes)
        classification_outputs_scores = tensorflow.python.saved_model.utils.build_tensor_info(values)
        classification_signature = tensorflow.python.saved_model.signature_def_utils.build_signature_def(
            inputs={tensorflow.python.saved_model.signature_constants.CLASSIFY_INPUTS: classification_inputs},
            outputs={
              tensorflow.python.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES: classification_outputs_classes,
              tensorflow.python.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES: classification_outputs_scores},
            method_name=tensorflow.python.saved_model.signature_constants.CLASSIFY_METHOD_NAME)
        tensor_info_x = tensorflow.python.saved_model.utils.build_tensor_info(x)
        tensor_info_y = tensorflow.python.saved_model.utils.build_tensor_info(y)
        prediction_signature = tensorflow.python.saved_model.signature_def_utils.build_signature_def(
            inputs={'images': tensor_info_x},
            outputs={'scores': tensor_info_y},
            method_name=tensorflow.python.saved_model.signature_constants.PREDICT_METHOD_NAME)
        legacy_init_op = tensorflow.group(tensorflow.tables_initializer(), name='legacy_init_op')
        builder.add_meta_graph_and_variables(
            sess, [tensorflow.python.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict_images': prediction_signature, 
                tensorflow.python.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: classification_signature},
            legacy_init_op=legacy_init_op)
        builder.save()
        print('Done exporting.')

if __name__ == '__main__':
    tensorflow.app.run()
