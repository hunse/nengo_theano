
import numpy as np
import numpy.random

import theano
import theano.tensor as T

from ensemble import Ensemble

class Connection(object):
    """A connection between two objects (Ensembles, Nodes, Networks)

    This class describes a connection between two objects. It contains
    the source (pre-population) and destination (post-population) of
    the connection. It also contains information about the computation of
    the connection, including functions or dimension transforms. Alternatively,
    it can represent a direct neuron-to-neuron connection by passing in a
    weight matrix. Finally, the connection can perform learning if given a
    learning rule.
    """

    def __init__(self, pre, post, neuron_space=False,
                 transform=None, function=None, weights=None,
                 filter=0.005, learning_rule=None):
        """
        Create a new connection between two objects. This connection should
        be added to a common parent of the objects.

        :param pre: pre-population object
        :param post: post-population object
        :param transform: vector-space transform matrix describing the mapping
            between the pre-population and post-population dimensions
        :type transform: a (pre.dimensions x post.dimensions) array of floats
        :param function: the vector-space function to be computed by the
            pre-population decoders
        :param weights: the connection-weight matrix for connecting pre-neurons
            to post-neurons directly. Cannot be used with transform or function.
        :type weights: a (pre.neurons x post.neurons) array of floats
        :param filter: a Filter object describing the post-synaptic filtering
            properties of the connection
        :param learning_rule: a LearningRule object describing the learning
            rule to use with the population
        """
        self._cache = {}

        if neuron_space and (transform is not None or function is not None):
            raise ValueError("Cannot provide \"function\" or \"transform\" in neuron space")
        elif not neuron_space and weights is not None:
            raise ValueError("Cannot provide \"weights\" in vector space")

        self.neuron_space = neuron_space

        # if neuron_space and (isinstance(pre, list) or isinstance(post,list)):
            # raise ValueError("Cannot do neuron-neuron connections with

        ### basic parameters, set by network.connect(...)
        self.pre = pre
        self.post = post
        self.transform = transform
        self.function = function
        self.weights = weights
        self.filter = filter
        self.learning_rule = learning_rule

        ### additional (advanced) parameters
        self._modulatory = False

        ### internal parameters
        self.output = None

    @property
    def modulatory(self):
        """Setting \"modulatory\" to True stops the connection from imparting
        current on the post-population."""
        return self._modulatory

    @modulatory.setter
    def modulatory(self, value):
        self._modulatory = value

    def build(self, dtype):
        self._cache.clear()
        self.dtype = dtype

        ### determine input
        self.spiking_input = False
        if isinstance(self.pre, list):
            all_ensembles = all(isinstance(pre, Ensemble) for pre in self.pre)
            no_ensembles = not any(isinstance(pre, Ensemble) for pre in self.pre)
            self.spiking_input = all_ensembles
            if self.neuron_space and all_ensembles:
                # self.input = np.concatenate([pre.neurons.spikes for pre in self.pre])
                self.input = T.concatenate([pre.neurons.spikes for pre in self.pre])
                self.pre_dims = sum(pre.neurons.size for pre in self.pre)
            elif all_ensembles:
                self.input = [pre.neurons.spikes for pre in self.pre]
                self.pre_dims = sum(pre.dimensions for pre in self.pre)
            elif no_ensembles:
                self.input = [pre.output for pre in self.pre]
                self.pre_dims = sum(pre.size for pre in self.pre)
            else:
                raise ValueError("\"pre\" list cannot mix Ensembles and Nodes")
        else:
            if isinstance(self.pre, Ensemble):
                self.input = self.pre.neurons.spikes
                self.pre_dims = self.pre.neurons.size
                self.spiking_input = True
            else:
                self.input = self.pre.output
                self.pre_dims = self.pre.size

        ### get dims (# of neurons for neuron_space, # of dimensions o.w.)
        if isinstance(self.post, list):
            post_dim_list = [el.get_size(neuron_space=self.neuron_space)
                             for el in self.post]
            self.post_dims = sum(post_dim_list)
        else:
            self.post_dims = self.post.get_size(neuron_space=self.neuron_space)

        ### make output storage
        if self.output is None:
            self.output = theano.shared(
                np.zeros(self.post_dims, dtype=dtype), name='output')

        ### convert transform or weight matrix
        if self.transform is not None:
            self.transform = np.asarray(self.transform, dtype=dtype)
            assert self.transform.shape == (self.pre_dims, self.post_dims)

        if self.weights is not None:
            self.weights = np.asarray(self.weights, dtype=dtype)
            assert self.weights.shape == (self.pre_dims, self.post_dims)

        ### notify the post object about the connection
        if isinstance(self.post, list):
            i = 0
            for post, dims in zip(self.post, post_dim_list):
                post.add_connection(self, self.output[i:i+dims])
                i += dims
        else:
            self.post.add_connection(self, self.output)

        ### vector-space connections from Ensembles need to make decoders
        if not self.neuron_space and isinstance(self.pre, list) and \
                all(isinstance(pre, Ensemble) for pre in self.pre):
            self.decoders = [pre.compute_decoders(self.function)
                             for pre in self.pre]
        elif not self.neuron_space and isinstance(self.pre, Ensemble):
            self.decoders = self.pre.compute_decoders(self.function)
        else:
            self.decoders = None

    def reset(self):
        self.output.set_value(0*self.output.get_value())
        # self.output.set_value(0)
        # self.output[:] = 0

    def tick(self, dt):
        if 'tick' not in self._cache:
            if self.neuron_space:
                if self.weights is not None:
                    value = T.dot(self.input, self.weights)
                else:
                    value = self.input
            elif isinstance(self.input, list):
                ### list of populations to decode
                if self.decoders is not None:
                    value = T.concatenate([T.dot(input, dec) for input, dec in
                                           zip(self.input, self.decoders)])
                else:
                    value = T.concatenate(self.input)

                if self.transform is not None:
                    value = T.dot(value, self.transform)
            else:
                ### Connect single nodes in vector-space.
                ### Decoders should not be None, unless connecting non-Ensembles
                if self.decoders is not None and self.transform is not None:
                    value = T.dot(T.dot(self.input, self.decoders), self.transform)
                elif self.decoders is not None:
                    value = T.dot(self.input, self.decoders)
                elif self.transform is not None:
                    value = T.dot(self.input, self.transform)
                else:
                    value = self.input

            ### filter output
            if self.filter > 0:
                decay = np.exp(-dt / self.filter).astype(self.dtype)
                if self.spiking_input:
                    output = decay*self.output + \
                        T.cast((1-decay)/dt, self.dtype)*value
                else:
                    output = decay*self.output + \
                        T.cast(1-decay, self.dtype)*value
            else:
                output = value

            self._cache['dt'] = dt
            self._cache['tick'] = theano.function(
                [], [], updates=[(self.output, output)])

        assert self._cache['dt'] == dt, "Cannot change 'dt' during simulation"
        self._cache['tick']()

    # def compute_transform(self, dim_pre, dim_post, array_size, weight=1,
    #                       index_pre=None, index_post=None, transform=None):
    #     """Helper function used by :func:`Network.connect()` to create
    #     the `dim_pre` by `dim_post` transform matrix.

    #     Values are either 0 or *weight*. *index_pre* and *index_post*
    #     are used to determine which values are non-zero, and indicate
    #     which dimensions of the pre-synaptic ensemble should be routed
    #     to which dimensions of the post-synaptic ensemble.

    #     :param int dim_pre: first dimension of transform matrix
    #     :param int dim_post: second dimension of transform matrix
    #     :param int array_size: size of the network array
    #     :param float weight: the non-zero value to put into the matrix
    #     :param index_pre: the indexes of the pre-synaptic dimensions to use
    #     :type index_pre: list of integers or a single integer
    #     :param index_post:
    #         the indexes of the post-synaptic dimensions to use
    #     :type index_post: list of integers or a single integer
    #     :returns:
    #         a two-dimensional transform matrix performing
    #         the requested routing

    #     """

    #     if transform is None:
    #         # create a matrix of zeros
    #         transform = [[0] * dim_pre for i in range(dim_post * array_size)]

    #         # default index_pre/post lists set up *weight* value
    #         # on diagonal of transform

    #         # if dim_post * array_size != dim_pre,
    #         # then values wrap around when edge hit
    #         if index_pre is None:
    #             index_pre = range(dim_pre)
    #         elif isinstance(index_pre, int):
    #             index_pre = [index_pre]
    #         if index_post is None:
    #             index_post = range(dim_post * array_size)
    #         elif isinstance(index_post, int):
    #             index_post = [index_post]

    #         for i in range(max(len(index_pre), len(index_post))):
    #             pre = index_pre[i % len(index_pre)]
    #             post = index_post[i % len(index_post)]
    #             transform[post][pre] = weight

    #     transform = np.array(transform)

    #     # reformulate to account for post.array_size
    #     if transform.shape == (dim_post * array_size, dim_pre):

    #         array_transform = [[[0] * dim_pre for i in range(dim_post)]
    #                            for j in range(array_size)]

    #         for i in range(array_size):
    #             for j in range(dim_post):
    #                 array_transform[i][j] = transform[i * dim_post + j]

    #         transform = array_transform

    #     return transform
