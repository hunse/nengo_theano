
import numpy as np
import theano
import theano.tensor as T

class Probe(object):

    def __init__(self, name, dimensions, dt=0.01, start_size=100):
        """
        :param string name:
        :param Network network:
        :param target:
        :type target:
        :param string target_name:
        :param float dt_sample:
        :param float pstc:
        """
        self._cache = {}
        self.name = name
        self.dimensions = dimensions
        self.dt = dt

        # create array to store the current input
        self.input = theano.shared(np.zeros(dimensions), name='input')
        self.connections = {}

        # create array to store the data over many time steps
        self.buffer = np.zeros((start_size, dimensions))
        self.reset()

    def build(self, dtype):
        self._cache.clear()

    def get_size(self, neuron_space):
        return self.dimensions

    def add_connection(self, connection, input):
        if connection not in self.connections:
            self.connections[connection] = input

    def reset(self):
        # self.input[:] = 0
        # self.input.set_value(0)
        self.input.set_value(0*self.input.get_value())
        self.buffer[:] = 0
        self.t = 0
        self.i = 0

    @property
    def max_time(self):
        return (len(self.buffer) - 1)*self.dt

    @max_time.setter
    def max_time(self, time):
        n = int(np.round(time/self.dt) + 1)
        if n > len(self.buffer):
            oldbuffer = self.buffer
            self.buffer = np.zeros((n, self.buffer.shape[1]))
            self.buffer[:len(oldbuffer)] = oldbuffer

    def run(self, t_end):
        while self.t + self.dt <= t_end:
            self.t += self.dt
            self.i += 1

            if self.i >= len(self.buffer):
                # out of buffer space, so double the buffer size
                self.max_time = 2*self.t

            self.buffer[self.i,:] = self.input.get_value()

    def tick(self, dt):
        if 'tick' not in self._cache:
            target = T.zeros_like(self.input)
            for input in self.connections.values():
                target += input

            self._cache['tick'] = theano.function(
                [], [], updates=[(self.input, target)])

        self._cache['tick']()

    @property
    def data(self):
        return self.buffer[:self.i+1]

