"""Paleo"""

__version__ = '0.1'

#
# The following environment variables are set to disable TensorFlow's software
# specific optimizations.
# This is because we want to check whether our implementation matches the
# behavior of underlying cuDNN library.
#

import os
# Disable TensorFlow's specific optimizations.
os.environ['TF_USE_DEEP_CONV2D'] = '0'

# Disable TensorFlow's autotuning mechanism, always fall back to cuDNN
# heuristics.
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'

# Set TF working space to 4 GB, as in TF's default setting.
os.environ['TF_CUDNN_WORKSPACE_LIMIT_IN_MB'] = '4096'
