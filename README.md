# Google Brain Speech Recognition Competition

Written for ID2223 at KTH by
Marc Jourdes
Eric Ren
Daniel Schlaug

Uses Tensorflow 1.4 and Python 2.7 (because Hops, our cluster needs pydoop which does not support Python 3).

## Preparing the data
The data needs to be downloaded and converted to tfrecords. Unzip the data in the data folder and run `python data/tfrecords_writer.py` from the project root.