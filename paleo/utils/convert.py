"""Test convertors"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json

from paleo.utils.convertors import CaffeConvertor

if __name__ == '__main__':
    filename = sys.argv[1]

    convertor = CaffeConvertor()
    net = convertor.convert(filename)
    print(json.dumps(net.as_dict(), sort_keys=True, indent=4))
