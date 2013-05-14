
import theano

from network import Network

class Model(object):

    def __init__(self, name, dt=0.001, dtype=theano.config.floatX):

        self.name = name
        self.dtype = dtype
        # self.network = network
        self.network = Network(self.name)

        self.t = 0
        self.dt = dt

        self.neuron_models = []
        self.connections = []

    def reset(self):
        self.t = 0
        for model in self.neuron_models:
            model.reset()
        for connection in self.connections:
            connection.reset()
        self.network.reset()

    def build(self):
        ensembles = self.network.all_ensembles

        ##################################################
        ### build all ensembles
        for ensemble in ensembles:
            ensemble.build(self.dtype)

        ##################################################
        ### make one big set of neurons for each neuron type
        kinds = set(type(e.neurons) for e in ensembles)
        # assert len(kinds) == 1 and kinds[0] == 'lif'
        # neuron_model = type(ensembles[0].neurons)

        self.neuron_models = []

        for kind in kinds:

            ### count number of neurons
            neurons = sum(e.neurons.size
                          for e in ensembles if isinstance(e.neurons, kind))

            big_model = kind(neurons, mode='bigmodel')
            big_model.build(self.dtype)

            i = 0
            for ensemble in ensembles:
                big_model.link_model(ensemble.neurons, i)
                i += ensemble.neurons.size

            self.neuron_models.append(big_model)


        ##################################################
        ### Build other network components (inputs, probes, etc.)
        self.network.build(self.dtype)

        ##################################################
        ### TODO: run neuron models here to estimate rates,
        ### then use these rates to find decoders

        ##################################################
        self.connections = self.network.all_connections
        for connection in self.connections:
            connection.build(self.dtype)


    def run(self, time):

        if len(self.neuron_models) == 0:
            raise Exception(
                "Either no neurons have been added to the model, " +
                "or the model has not been built yet.")

        stop_time = self.t + time

        ### get probes to make buffers for the full runtime (it's faster)
        for probe in self.network.all_probes:
            if 2*probe.max_time < stop_time:
                probe.max_time = stop_time

        while self.t < stop_time:
            self.t += self.dt

            ### reset inputs; connections will write to these inputs
            # self.network.reset_input()
            # for model in self.neuron_models:
            #     model.reset_input()

            ### update connections
            for connection in self.connections:
                connection.tick(self.dt)

            ### call ticks, this is where inputs get written from connections
            self.network.tick(self.dt)

            ### run neuron models
            for model in self.neuron_models:
                model.run(self.t)

            ### run everything else (i.e., nodes and probes)
            self.network.run(self.t)
