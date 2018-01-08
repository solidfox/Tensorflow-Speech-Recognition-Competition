"""Package containing deep neural network models for classifying the speech.

Since the models should be deployable to a Raspberry Pi 3 with 1 GB RAM one might assume a limit of 265 MBs of weights.
With each weight at 32bits = 4bytes we are constrained to 64 million weights. Meaning approximately
"""

__author__ = 'Daniel Schlaug'

from convolutional import convolutional_model_fn
