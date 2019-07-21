import numpy as np
import time
import sys
import six

def get_from_module(identifier, module_params, module_name,
                    instantiate=False, kwargs=None):
    if isinstance(identifier, six.string_types):
        res = module_params.get(identifier)
        if not res:
            raise Exception('Invalid ' + str(module_name) + ': ' +
                            str(identifier))
        if instantiate and not kwargs:
            return res()
        elif instantiate and kwargs:
            return res(**kwargs)
        else:
            return res
    elif type(identifier) is dict:
        name = identifier.pop('name')
        res = module_params.get(name)
        if res:
            return res(**identifier)
        else:
            raise Exception('Invalid ' + str(module_name) + ': ' +
                            str(identifier))
    return identifier
