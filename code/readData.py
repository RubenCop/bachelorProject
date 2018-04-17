from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

IMAGE_ZIZE = 24
NUM_CLASSES = 10 #test
EXAMPLES_TRAIN_EPOCH = 50000 #NOT KNOWN YET
EXAMPLES_EVAL_EPOCH = 10000 #NOT KNOWN YET

def read_data(filename_queue):
    class dataRecord(object):
        pass
    result = dataRecord()

    label_byes = 1
    result.height = 32
    result.width = 32
    result.depth = 1
    image_bytes = result.height * result.width * result.depth

    record_bytes = label_bytes + image_bytes
    