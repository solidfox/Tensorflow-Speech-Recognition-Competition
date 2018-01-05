# Save model
# TODO: https://www.tensorflow.org/programmers_guide/saved_model#models
# https://stackoverflow.com/questions/45705070/how-to-load-and-use-a-saved-model-on-tensorflow

import tensorflow as tf


def save_model(input_data, model):
    """
    :param input_data: input data
    :param model: The model that takes the input data
    :return:
    """
    export_dir = "/model/saved"
    signatures = {
        "model": tf.saved_model.signature_def_utils.predict_signature_def(
            inputs={"input": input_data},
            outputs={"model": model})
    }
    builder = tf.saved_model_builder.SavedModelBuilder(export_dir)
    with tf.Session(graph=tf.Graph()) as sess:
        builder.add_meta_graph_and_variables(sess,
                                             ["finalModel"],
                                             signature_def_map=signatures)
    builder.save()