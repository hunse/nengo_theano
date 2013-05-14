
from ensemble import Ensemble
from connection import Connection
from probe import Probe
from input import Input
from neurons import LIFNeuronModel

class Network(object):
    def __init__(self, name):

        self.name = name
        self.Connections = []
        self.Ensembles = []
        self.Networks = []
        self.Nodes = []
        self.Probes = []

        self.reset()

    def connect(self, pre, post, function=None, transform=None, scale=None,
                filter=0.005, learning_rule=None):

        connection = Connection(pre, post, neuron_space=False,
                                function=function, transform=transform,
                                scale=scale,
                                filter=filter, learning_rule=learning_rule)
        self.Connections.append(connection)
        return connection

    def connect_neurons(self, pre, post, weights=None, scale=None,
                        filter=0.005, learning_rule=None):

        connection = Connection(pre, post, neuron_space=True,
                                weights=weights, scale=scale,
                                filter=filter, learning_rule=learning_rule)
        self.Connections.append(connection)
        return connection

    def make_ensemble(self, name, neurons, dimensions=1,
                      max_rate=(50,100), intercept=(-1,1), radius=1.0,
                      encoders=None, neuron_model=dict(type=LIFNeuronModel,
                      tau_ref=0.002, tau_rc=0.02), mode='spiking'):

        ensemble = Ensemble(name, neurons, dimensions,
                            max_rate=max_rate, intercept=intercept, radius=radius,
                            encoders=encoders, neuron_model=neuron_model,
                            mode=mode)

        self.Ensembles.append(ensemble)
        return ensemble

    def make_input(self, name, value):
        input = Input(name, value)
        self.Nodes.append(input)
        return input

    def make_probe(self, name, dimensions, dt=0.01):
        probe = Probe(name, dimensions, dt=dt)
        self.Probes.append(probe)
        return probe

    @property
    def all_ensembles(self):
        ensembles = list(self.Ensembles)
        for network in self.Networks:
            ensembles.extend(network.all_ensembles)
        return ensembles

    @property
    def all_connections(self):
        connections = list(self.Connections)
        for network in self.Networks:
            connections.extend(network.all_connections)
        return connections

    @property
    def all_probes(self):
        probes = list(self.Probes)
        for network in self.Networks:
            probes.extend(network.all_probes)
        return probes

    @property
    def objects(self):
        return self.Connections + self.Ensembles + self.Networks + \
            self.Nodes + self.Probes

    def build(self, dtype):
        for network in self.Networks:
            network.build(dtype)

        for node in self.Nodes:
            node.build(dtype)

        for probe in self.Probes:
            probe.build(dtype)

    def reset(self):
        for obj in self.objects:
            obj.reset()

    def run(self, t_end):
        for network in self.Networks:
            network.run(t_end)

        for node in self.Nodes:
            node.run(t_end)

        for probe in self.Probes:
            probe.run(t_end)

    def tick(self, dt):
        for network in self.Networks:
            network.tick(dt)

        for ensemble in self.Ensembles:
            ensemble.tick(dt)

        # for node in self.Nodes:
            # node.tick(dt)

        for probe in self.Probes:
            probe.tick(dt)
