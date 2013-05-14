
import numpy as np
import numpy.random
import numpy.linalg

import theano
import theano.tensor as T

from neurons import LIFNeuronModel

import ipdb

class Ensemble(object):
    # def __init__(self, name, neurons, max_rate, intercept,
    #              alpha, bias, tau_rc, tau_ref):

    def __init__(self, name, neurons, dimensions,
                 max_rate, intercept, radius,
                 encoders, neuron_model, mode):

        self._cache = {}
        self.name = name

        neuron_model_type = neuron_model.pop('type', LIFNeuronModel)
        self.neurons = neuron_model_type(neurons, mode=mode, **neuron_model)

        if max_rate is not None and intercept is not None:
            self.neurons.set_rate_intercept(
                max_rate=max_rate, intercept=intercept, radius=radius)
        elif max_rate is not None or intercept is not None:
            raise ValueError("Must set both \"max_rate\" and \"intercept\", or neither")

        self.dimensions = dimensions
        self.encoders = encoders
        self.radius = radius

        self.vector_inputs = {} # incoming vector-space connections
        self.neuron_inputs = {} # incoming neuron-space connections

        # self.neurons = LIFNeuronModel(neurons,
        #                               max_rate=max_rate, intercept=intercept,
        #                               alpha=alpha, bias=bias,
        #                               tau_rc=tau_rc, tau_ref=tau_ref)
        # self.reset()

    @property
    def alpha(self):
        return self.neurons.alpha

    @property
    def bias(self):
        return self.neurons.bias

    def get_size(self, neuron_space=False):
        if neuron_space:
            return self.neurons.size
        else:
            return self.dimensions

    def add_connection(self, connection, input):
        if connection.neuron_space:
            if connection not in self.neuron_inputs:
                self.neuron_inputs[connection] = input
        else:
            if connection not in self.vector_inputs:
                self.vector_inputs[connection] = input

    def build(self, dtype):
        self._cache.clear()
        self.dtype = dtype

        if self.encoders is None:
            ### pick encoders from multivariate normal distribution
            self.encoders = np.random.normal(
                size=(self.dimensions, self.neurons.size))
        else:
            self.encoders = np.asarray(self.encoders, dtype=dtype)
            assert self.encoders.shape == (self.dimensions, self.neurons.size)

        ### normalize so all encoders are unit vectors
        norm = np.sqrt(((self.encoders)**2).sum(axis=0))
        self.encoders = self.encoders / norm

        self.eval_points = None
        self._gamma_inv_mult = None
        self._a_matrix_mult = None

    def compute_decoders(self, function, noise=0.1):

        if self.eval_points is None:
            dims = self.dimensions
            n_samples = 500

            ##################################################
            ### generate sample points from a clipped normal distribution
            # self.eval_points = np.random.normal(
            #     size=(n_samples, dims), scale=0.5*self.radius)
            # norm = np.sqrt((self.eval_points**2).sum(axis=-1))
            # self.eval_points[norm > radius] /= norm[norm > radius]

            ### generate sample points in a hypersphere
            self.eval_points = np.random.normal(size=(n_samples, dims))

            # generate magnitudes for vectors from uniform distribution
            scale = np.random.uniform(size=n_samples, low=0, high=self.radius)
            norm = np.sqrt((self.eval_points**2).sum(axis=-1))

            self.eval_points *= (scale / norm)[:,None]

            ##################################################
            ### get neuron activations at eval_points

            inputs = np.dot(self.eval_points, self.encoders)
            if hasattr(self.neurons, "rate"):
                A = self.neurons.rate(inputs)
            else:
                raise NotImplementedError(
                    "TOOD: run neurons to estimate firing rate")

            ##################################################
            ### find and invert gamma matrix

            # sigmas = noise * A.max(axis=0)
            sigmas = noise * A.max()
            gamma = np.dot(A.T, A) # + np.diag(sigmas**2)
            np.fill_diagonal(gamma, gamma.diagonal() + sigmas**2)

            L = np.linalg.cholesky(gamma)
            L = np.linalg.inv(L.T)
            self._gamma_inv_mult = lambda x: np.dot(L, np.dot(L.T, x))
            self._a_matrix_mult = lambda x: np.dot(A.T, x)

        ##################################################
        ### determine function value at eval points
        if function is not None:
            y0 = function(self.eval_points[0])
            y = np.zeros((len(self.eval_points), len(y0)))
            for i, p in enumerate(self.eval_points):
                y[i] = function(p)
        else:
            y = self.eval_points

        decoders = self._gamma_inv_mult(self._a_matrix_mult(y))
        return decoders

    def reset(self):
        pass

    def reset_input(self):
        pass
        # self.neurons.input[:] = 0

    # @property
    # def output(self):
    #     return self.neurons.spikes

    # def run(self, t_end):
    #     pass

    def tick(self, dt):
        if 'tick' not in self._cache:
            # target = self.neurons.input
            target = T.zeros_like(self.neurons.input)

            ### get values from vector connections and multiply by encoders
            for input in self.vector_inputs.values():
                target = target + T.dot(input, T.cast(self.encoders, self.dtype))

            ### get values from neuron connections
            for input in self.neuron_inputs.values():
                target = target + input

            # if isinstance(self.neurons.input, theano.tensor.Subtensor):
            if isinstance(self.neurons.input, theano.tensor.basic.TensorVariable):
                updates = [(self.neurons.parent.input,
                            T.set_subtensor(self.neurons.input, target))]
            else:
                updates = [(self.neurons.input, target)]

            self._cache['tick'] = theano.function([], [], updates=updates)

        self._cache['tick']()

        # ### get values from vector connections and multiply by encoders
        # for input in self.vector_inputs.values():
        #     if 'add_encoded' not in self._cache:
        #         a = T.vector(dtype=self.neurons.dtype)
        #         b = self.neurons.input + T.dot(a, self.encoders)
        #         self._cache['add_encoded'] = theano.function(
        #             [a], [], updates=[(self.neurons.input, b)])

        #     self._cache['add_encoded'](input)
        #     # self.neurons.input += np.dot(input, self.encoders)

        # ### get values from neuron connections
        # for input in self.neuron_inputs.values():
        #     if 'add' not in self._cache:
        #         a = T.vector(dtype=self.neurons.dtype)
        #         b = self.neurons.input + a
        #         self._cache['add'] = theano.function(
        #             [a], [], updates=[(self.neurons.input, b)])

        #     self._cache['add'](input)
        #     # self.neurons.input += input
