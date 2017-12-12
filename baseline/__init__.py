__author__ = 'Daniel Schlaug'

import os
import tensorflow as tf
from data_loading import *
from data_iterator import *
from model_handling import *
from network_configuration import *
from generate_submission import *

def main():
    params = dict(
        seed=2018,
        batch_size=64,
        keep_prob=0.5,
        learning_rate=1e-3,
        clip_gradients=15.0,
        use_batch_norm=True,
        num_classes=len(POSSIBLE_LABELS),
    )

    hparams = tf.contrib.training.HParams(**params)
    os.makedirs(os.path.join(OUTDIR, 'eval'))
    model_dir = OUTDIR

    run_config = tf.contrib.learn.RunConfig(model_dir=model_dir)

    # it's a magic function :)
    from tensorflow.contrib.learn.python.learn.learn_io.generator_io import generator_input_fn

    train_input_fn = generator_input_fn(
        x=data_generator(trainset, hparams, 'train'),
        target_key='target',  # you could leave target_key in features, so labels in model_handler will be empty
        batch_size=hparams.batch_size, shuffle=True, num_epochs=None,
        queue_capacity=3 * hparams.batch_size + 10, num_threads=1,
    )

    val_input_fn = generator_input_fn(
        x=data_generator(valset, hparams, 'val'),
        target_key='target',
        batch_size=hparams.batch_size, shuffle=True, num_epochs=None,
        queue_capacity=3 * hparams.batch_size + 10, num_threads=1,
    )

    def _create_my_experiment(run_config, hparams):
        exp = tf.contrib.learn.Experiment(
            estimator=create_model(config=run_config, hparams=hparams),
            train_input_fn=train_input_fn,
            eval_input_fn=val_input_fn,
            train_steps=10000,  # just randomly selected params
            eval_steps=200,  # read source code for steps-epochs ariphmetics
            train_steps_per_iteration=1000,
        )
        return exp

    tf.contrib.learn.learn_runner.run(
        experiment_fn=_create_my_experiment,
        run_config=run_config,
        schedule="continuous_train_and_eval",
        hparams=hparams)

    generate_submission(wavfile=wavfile,
                        datadir=DATADIR,
                        model_dir=model_dir,
                        run_config=run_config,
                        hparams=hparams)


if __name__ == '__main__':
    main()
