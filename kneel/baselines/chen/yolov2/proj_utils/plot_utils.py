try:
    from visdom import Visdom
except:
    print('Better install visdom')
import numpy as np
import random

import scipy.misc
from scipy.misc import imsave

from .local_utils import imshow, writeImg, normalize_img
_port = 8899



class plot_scalar(object):
    def __init__(self, name='default', env='main', rate= 1, handler=None, port = _port):

        self.__dict__.update(locals())
        self.values = []
        self.steps = []
        if self.handler is None:
            self.handler = Visdom(port=port)
        self.count = 0

    def plot(self,values, step = None):
        org_type_chk = type(values) is  list
        if not org_type_chk:
            values = [values]

        len_val = len(values)
        if step is None:
            step = list(range(self.count, self.count+len_val))

        self.count += len_val
        self.steps.extend(step)
        self.values.extend(values)

        if self.count % self.rate == 0 or org_type_chk:
            self.flush()

    def reset(self):
        self.steps = []
        self.values = []

    def flush(self):
        #print('flush the plot. :)')
        assert type(self.values) is list, 'values have to be list'
        if type(self.values[0]) is not list:
            self.values = [self.values]

        n_lines = len(self.values)
        repeat_steps = [self.steps]*n_lines
        steps  = np.array(repeat_steps).transpose()
        values = np.array(self.values).transpose()

        assert not np.isnan(values).any(), 'nan error in loss!!!'
        res = self.handler.line(
                X = steps,
                Y=  values,
                win= self.name,
                update='append',
                opts=dict(title = self.name),
                env = self.env
            )

        if res != self.name:
            self.handler.line(
                X=steps,
                Y=values,
                win=self.name,
                env=self.env,
                opts=dict(title=self.name)
            )

        self.reset()
