import tensorflow as tf


def get_input_fn(data, labels, target_column, mode):
    if mode == 'train':
        return tf.estimator.inputs.pandas_input_fn(data, y=labels, batch_size=32, num_epochs=None, shuffle=True, target_column=target_column)
    else:
        return tf.estimator.inputs.pandas_input_fn(data, y=labels, batch_size=32, num_epochs=1, shuffle=False, target_column=target_column)


def get_export_fn(features):
    inputs = {}

    for feature in features:
        inputs[feature.name] = tf.placeholder(shape=[None], dtype=feature.dtype)

    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


def get_estimator(features, model_dir):

    return tf.estimator.LinearRegressor(features, model_dir=model_dir, optimizer=tf.train.FtrlOptimizer(learning_rate=1, l1_regularization_strength=0.001))

