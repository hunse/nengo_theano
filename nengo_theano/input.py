
import numpy as np

import theano

# class Input(object):
#     def __init__(self, dimensions, filter=None):
#         self.set([0] * dimensions)
#         self.filter = filter
#     def update(self, dt):
#         if self.filter is not None:
#             self.state = self.filter.update(self.raw, dt)
#         else:
#             self.state = self.raw
#     def set(self, value):
#         self.raw = np.array(value)
#     def get(self):
#         return self.state
#     def reset(self):
#         self.set([0] * len(self.state))

def clean_value(value, dtype=None):
    if dtype is None:
        return np.asarray(value).flatten()
    else:
        return np.asarray(value, dtype=dtype).flatten()

class Input(object):

    def __init__(self, name, value):

        self.name = name
        self.change_time = None
        self.function = None

        if callable(value):
            ### if value parameter is a python function
            self.function = lambda t: clean_value(value(t))
            self._value_0 = self.function(0)
        elif isinstance(value, dict):
            ### if value is dict of time:value pairs
            self.keys = sorted(value.keys())
            self.dict = \
                dict((k, clean_value(value[k])) for k in self.keys)
            self.change_time = self.keys[0]
            self._value_0 = self.dict[self.change_time]
        else:
            self._value_0 = clean_value(value)

    def build(self, dtype):
        self.dtype = dtype
        self._size = self._value_0.size
        self._value = theano.shared(self._value_0.astype(dtype), name='value')

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, x):
        # self._value[:] = x
        self._value.set_value(clean_value(x, dtype=self.dtype))

    @property
    def size(self):
        return self._size

    @property
    def output(self):
        return self._value

    def reset(self):
        if self.function is not None:
            self.value = self.function(0)
        elif self.change_time is not None:
            self.change_time = self.keys[0]
            self.value = self.dict[self.change_time]

    def run(self, t_end):
        if self.function is not None:
            self.value = self.function(t_end)
        elif self.change_time is not None and t_end > self.change_time:
            self.value = self.dict[self.change_time]

            ### set change_time to next time after t_end, None if DNE
            self.change_time = next((t for t in self.keys if t > t_end), None)
