
import numpy as np
import numpy.random

import theano
import theano.tensor as T

def dist_or_list(arg, size):
    if isinstance(arg, tuple):
        ### distribution, generate random uniforms
        return numpy.random.uniform(size=size, low=arg[0], high=arg[1])
    else:
        ### list or list-like, make sure it's an array
        return np.asarray(arg)


class LIFNeuronModel(object):

    dt = 0.001

    def __init__(self, n_neurons, mode, tau_rc=0.02, tau_ref=0.002):

        self._cache = {}

        self.size = n_neurons
        self.mode = mode
        assert mode in ['spiking', 'bigmodel'],\
            "Non-spiking neurons not yet implemented"

        self.alpha = None
        self.bias = None

        self.tau_rc = tau_rc
        self.tau_ref = tau_ref

        self.v = None
        self.w = None

        self.t = 0

    def rate_intercept_to_alpha_bias(self, max_rate, intercept, radius):
        """Compute the alpha and bias needed to get the given max_rate
        and intercept values.

        Returns gain (alpha) and offset (j_bias) values of neurons.

        :param float array max_rates: maximum firing rates of neurons
        :param float array intercepts: x-intercepts of neurons
        """

        max_rates = dist_or_list(max_rate, self.size)
        intercepts = dist_or_list(intercept, self.size)

        x1 = intercepts
        x2 = radius
        z1 = 1.
        z2 = 1. / (1 - np.exp((self.tau_ref - 1./max_rates)/self.tau_rc))
        alpha = (z1 - z2) / (x1 - x2)
        bias = z1 - alpha * x1

        return alpha, bias

    def set_rate_intercept(self, max_rate, intercept, radius):
        self.alpha, self.bias = self.rate_intercept_to_alpha_bias(
            max_rate=max_rate, intercept=intercept, radius=radius)

    def rate(self, inputs):
        """Analytically compute the firing rates for constant input values."""
        # j = np.maximum(self.alpha*inputs + self.bias - 1, 0.0)
        # r = 1. / (self.tau_ref + self.tau_rc*np.log1p(1./j))

        j = self.alpha*inputs + self.bias - 1.
        r = np.zeros_like(j)
        r[j > 0] = 1. / (self.tau_ref + self.tau_rc*np.log1p(1./j[j > 0]))
        return r

    def reset(self):
        self._cache.clear()
        self.t = 0
        if self.v is not None: self.v.set_value(0*self.v.get_value())
        if self.w is not None: self.w.set_value(0*self.v.get_value())

    def build(self, dtype):
        self.dtype = dtype
        self.v = theano.shared(np.zeros(self.size, dtype=dtype), name='v')
        self.w = theano.shared(np.zeros(self.size, dtype=dtype), name='w')
        self.input = theano.shared(np.zeros(self.size, dtype=dtype), name='input')
        self.spikes = theano.shared(np.zeros(self.size, dtype=dtype), name='spikes')

        self.alpha = np.zeros(self.size, dtype=dtype)
        self.bias = np.zeros(self.size, dtype=dtype)

        self.tau_rc = np.zeros(self.size, dtype=dtype)
        self.tau_ref = np.zeros(self.size, dtype=dtype)

    def link_model(self, model, i):
        """Link a sub-model into this model at position 'i'"""

        assert isinstance(model, LIFNeuronModel), "Model must be the same type"
        n = model.size

        ### copy over parameters from sub-model
        # if model.v is not None: self.v[i:i+n] = model.v
        # if model.w is not None: self.w[i:i+n] = model.w

        self.alpha[i:i+n] = model.alpha
        self.bias[i:i+n] = model.bias
        self.tau_rc[i:i+n] = model.tau_rc
        self.tau_ref[i:i+n] = model.tau_ref

        ### link parameters back to sub-model
        model.input = self.input[i:i+n]
        model.spikes = self.spikes[i:i+n]
        model.v = self.v[i:i+n]
        model.w = self.w[i:i+n]
        model.parent = self

    def run(self, t_end):
        if 'step' not in self._cache:

            ### create run function
            dV = (self.dt/self.tau_rc) * (self.alpha*self.input + self.bias - self.v)

            ### increase the voltage, ignore values below 0
            v = T.maximum(self.v + dV, 0)

            ### Eric's method: overshoot approximation in dt units
            ### handle refractory period
            w = self.w - 1
            v = v * (1 - w).clip(0., 1.)

            ### determine which neurons spike
            spike = v > 1
            spikes = self.spikes + spike

            ### linearly approximate time since neuron crossed spike threshold
            overshoot = (v - 1) / dV
            w = T.switch(spike, self.tau_ref/self.dt - overshoot + 1.5, w)
            ### EH: adding 1.5 seems to have empirical benefit (matches analytic curve better)

            updates = [(self.v, v), (self.w, w), (self.spikes, spikes)]
            self._cache['step'] = theano.function([], [], updates=updates)

        self.spikes.set_value(0*self.spikes.get_value())
        while self.t < t_end - 0.5*self.dt:
            self.t += self.dt
            self._cache['step']()


    # def run(self, t_end):
    #     if 'run' not in self._cache:

    #         k = T.scalar(name='k_steps', dtype='int32')

    #         def step(v, w, spikes):

    #             ### create run function
    #             dV = (self.dt/self.tau_rc) * (self.alpha*self.input + self.bias - v)

    #             ### increase the voltage, ignore values below 0
    #             v = T.maximum(v + dV, 0)

    #             ### Eric's method: overshoot approximation in dt units
    #             ### handle refractory period
    #             w = w - 1
    #             v = v * (1 - w).clip(0., 1.)

    #             ### determine which neurons spike
    #             spike = v > 1
    #             spikes = spikes + spike

    #             ### linearly approximate time since neuron crossed spike threshold
    #             overshoot = (v - 1) / dV
    #             w = T.switch(spike, self.tau_ref/self.dt - overshoot + 1.5, w)
    #             # self.w[spike] = (self.tau_ref/self.dt - overshoot + 1.5)[spike]
    #             ### EH: adding 1.5 seems to have empirical benefit (matches analytic curve better)

    #             return v, w, spikes

    #         spikes = T.zeros_like(self.spikes)
    #         result, updates = theano.scan(
    #             fn=step, outputs_info=[self.v, self.w, spikes], n_steps=k)

    #         v, w, spikes = result
    #         updates.update([(self.v, v[-1]), (self.w, w[-1]), (self.spikes, spikes[-1])])

    #         self._cache['run'] = theano.function([k], [], updates=updates)

    #     # self.spikes[:] = 0
    #     # self.spikes.set_value(0)

    #     k = int(np.round((t_end - self.t) / self.dt))
    #     self._cache['run'](k)
    #     self.t += k*self.dt

    #     # while self.t < t_end - 0.5*self.dt:
    #     #     self.t += self.dt
    #     #     self._cache['step']()
