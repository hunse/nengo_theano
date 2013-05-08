"""
Test a basic communication channel between two populations
"""

import collections
import time

import numpy as np
import numpy.random as npr
import scipy.interpolate

import matplotlib
import matplotlib.pyplot as plt

import nengo_theano as nengo

##################################################
### create input

n_switch = 10
dt_switch = 0.8
dt_sample = 0.01

x = np.random.uniform(size=n_switch, low=-1, high=1)
tx = dt_switch*np.arange(len(x))
t_final = dt_switch*(len(x) + 1)

##################################################
### make the communication channel

model = nengo.Model('test')
net = model.network

input = net.make_input('input', dict(zip(tx,x)))

e1 = net.make_ensemble('e1', 100)
e2 = net.make_ensemble('e2', 100)

probe = net.make_probe('probe', 1, dt=dt_sample)

net.connect(input, e1, filter=0.005)
net.connect(e1, e2, filter=0.005)
net.connect(e2, probe, filter=0.03)

timer = time.time()
model.build()
print "Build: ", time.time() - timer, "seconds"

##################################################
### run the model and plot the results

timer = time.time()
model.run(t_final)
print "Run: ", time.time() - timer, "seconds"

# model.reset()

# timer = time.time()
# model.run(t_final)
# print "Run2: ", time.time() - timer, "seconds"

y = probe.data
ty = dt_sample*np.arange(len(y))
u = scipy.interpolate.interp1d(
    tx, x, kind='zero', bounds_error=False, fill_value=x[-1])(ty)

plt.figure(4)
plt.clf()
plt.plot(ty, u, '--')
plt.plot(ty, y)
plt.show()
