import torch
import torch.nn as nn
import numpy as np
import math
import deepxde as dde
import os
from functools import reduce
from timeit import default_timer
import yaml
import warnings
warnings.simplefilter("ignore", UserWarning)
from datetime import datetime
import matplotlib.pyplot as plt

def read_yaml(file_path):           # To read the configurations file
    with open(file_path, "r") as f:
        return yaml.safe_load(f)
#-----------Solving a SVE: for data creation----------------------------
class SVE():
    def __init__(self, g, K_mu, K_sigma, mu, sigma):
        self.g = g
        self.K_mu = K_mu
        self.K_sigma = K_sigma
        self.mu = mu
        self.sigma = sigma

    def solver_(self, B, X_0, T, dt):       # solve an SVE using Euler-Maruyama-scheme
        if T/dt > int(T/dt): raise ValueError("dt is not of the form 1/m")
        Solution = np.zeros(shape = B.shape)
        X_0 = torch.from_numpy(X_0)
        Solution[0, :, :] = X_0*self.g(0)
        
        # Euler-Maruyama-method for SVEs, see [Zhang2008]:
        Solution[1,:,:] = X_0*self.g(dt) + self.K_mu(0, dt) * self.mu(0, X_0) * dt + self.K_sigma(0, dt) * self.sigma(0, X_0)* B[1,:,:]  # first step in the E.-M.-scheme
        for i in range(1, int(T/dt)):
            # we calculate the value of the integrals in the E.-M.-scheme formula
            int_val = sum([self.K_mu(j*dt, (i+1)*dt) * self.mu(j*dt, Solution[j]) * dt + self.K_sigma(j*dt, (i+1)*dt) * self.sigma(j*dt, Solution[j]) * (B[j+1,:,:] - B[j,:,:]) for j in range(1,i+1)])
            Solution[i+1,:,:] = X_0*self.g((i+1)*dt) + int_val + self.K_mu(0,(i+1)*dt) * self.mu(0,X_0)*dt + self.K_sigma(0,(i+1)*dt) * self.sigma(0,X_0) * B[1,:,:]
        return Solution
#-----Data generating-------------------------------
class Noise(object):    
    def partition(self, a,b, dx): #makes a partition of [a,b] of equal sizes dx
        return np.linspace(a, b, int((b - a) / dx) + 1)
    # Create l dimensional Brownian motion with time step = dt
    def BM(self, start, stop, dt, l, dim):
        T = self.partition(start, stop, dt)
        # assign to each point of len(T) time point an N(0, \sqrt(dt)) standard l dimensional random variable
        BM = np.random.normal(scale=np.sqrt(dt), size=(len(T), l, dim))
        BM[0] = 0 #set the initial value to 0
        BM = np.cumsum(BM, axis  = 0) # cumulative sum: B_n = \sum_1^n N(0, \sqrt(dt))
        return BM
    
def data_generator(n, ntrain, dim, T, dt, sve_type, epsilon, theta_o, alpha, theta, kappa, xi, x0):
    X_0 = x0*np.ones([n, dim])  # Initial value
    B = Noise().BM(0, T, dt, n, dim)     # Brownian motion realizations
    # Generate SVE parameters dependent on sve_type
    if sve_type == "PEN":
        g = lambda t: 1
        K_mu = lambda s, t: s-t
        K_sigma = lambda s, t: epsilon*(s-t)
        mu = lambda s, x: x
        sigma = lambda s,x: x
    elif sve_type == "OU":          # TODO
        g = lambda t: np.exp(-theta_o*t)
        K_mu = lambda s, t: np.exp(-theta_o*(t-s))
        K_sigma = lambda s, t: np.exp(-theta_o*(t-s))
        mu = lambda s, x: x
        sigma = lambda s,x: np.sqrt(np.abs(x))
    elif sve_type == "RH":
        g = lambda t: 1
        K_mu = lambda s, t: np.power((t-s), -alpha)
        K_sigma = lambda s, t: np.power((t-s), -alpha)
        mu = lambda s, x: kappa*(theta-x)/math.gamma(1-alpha)
        sigma = lambda s,x: xi * np.sqrt(np.abs(x))/math.gamma(1-alpha)

    # solve the SVE
    Sol = SVE(g, K_mu, K_sigma, mu, sigma).solver_(B, X_0, T, dt)
    #---------------------------------------------------
    B_t = np.transpose(B.astype(np.float32)[1:,:,0])  # numpy arrays
    data = np.transpose(Sol.astype(np.float32)[1:,:,0])
    #---------------------------------------------------------------------------------------------------------------------
    X_train = (B_t[:ntrain,:], dt + np.expand_dims(np.arange(0,T,dt),1).astype(np.float32))
    y_train = data[:ntrain,:]
    X_test = (B_t[ntrain:,:], dt + np.expand_dims(np.arange(0,T,dt),1).astype(np.float32))
    y_test = data[ntrain:,:]
    data = dde.data.TripleCartesianProd(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    return data, X_train, y_train, X_test, y_test


def network_builder(T, dt, width, depth_branch, depth_trunk, mul_branch, mul_trunk, activation):
    branch_net = [int(T/dt)] + depth_branch*[width]
    branch_net[-2] = branch_net[-2]*(1+mul_branch) # double depth of the last hidden layer if mul_branch = True
    trunk_net = [dim] + depth_trunk * [width]
    trunk_net[-2] = trunk_net[-2]*(1+mul_trunk) # double depth of the last hidden layer if mul_trunk = True
    net = dde.nn.DeepONetCartesianProd(
        branch_net,  # width of the last layers must be equal
        trunk_net,
        activation,
        "Glorot normal",
    )
    return net

if __name__ == "__main__":
    #-------Loading the parameters-----------------------------------------
    params = read_yaml("configurations_deepONet.yaml") # Load the parameters
    width, depth_branch, depth_trunk, mul_branch, mul_trunk, activation, epochs, batch_size, learning_rate, scheduler_gamma, n, dim, T, dt, print_every, sve_type, epsilon, theta_o, mu, sigma, alpha, kappa, theta, xi, x0 = params.values()
    if dim != 1: raise ValueError("Only dim=1 implemented")
    ntrain=int(n*0.8)   # Derived parameters
    ntest=int(n*0.2)
    scheduler_step=int(epochs/4)+1
    #---------Foldername-----------------------------------
    absolute_path = os.getcwd()
    now = str(datetime.now()).replace(" ", "-").replace(":",".")   # actual time for the naming of the folder
    foldername = os.path.join(absolute_path, "DeepONet_{}_T{}_dt{}_epochs{}_n{}_time{}".format(sve_type,T, dt, epochs, n, now))
    os.makedirs(foldername)

    data, X_train, y_train, X_test, y_test = data_generator(n, ntrain, dim, T, dt, sve_type, epsilon, theta_o, alpha, theta, kappa, xi, x0)

    # Define a Model
    net = network_builder(T, dt, width, depth_branch, depth_trunk, mul_branch, mul_trunk, activation)
    model = dde.Model(data, net)

    # Compile and Train
    model.compile("adam", lr=learning_rate, loss="mean l2 relative error", decay = ("step",scheduler_step, scheduler_gamma))
    losshistory, train_state = model.train(iterations=epochs, batch_size=batch_size, display_every=print_every)
    dde.utils.plot_loss_history(losshistory)
    plt.savefig(os.path.join(foldername, "losshistory.png"))
    plt.clf()

    y = np.linspace(dt,int(T),int(T/dt))
    fig, ax = plt.subplots(2,int(5),figsize=(20,10))
    for i in range(int(10)):
        if i<=4:
            ax[0][i].plot(y, y_train[i,:], label="True Volterra path")
            ax[0][i].plot(y, model.predict((np.expand_dims(X_train[0][i,:],0),X_train[1])).squeeze(0), label="Estimated Volterra path")
        else:
            ax[1][i-5].plot(y, y_train[i,:], label="True Volterra path")
            ax[1][i-5].plot(y, model.predict((np.expand_dims(X_train[0][i,:],0),X_train[1])).squeeze(0), label="Estimated Volterra path")
    plt.suptitle(r'10 paths from the training set: n={}, T={}, dt={}'.format(n,T,dt))
    plt.legend()
    plt.savefig(os.path.join(foldername, "train_paths.png"))
    plt.clf()

    fig2, ax2 = plt.subplots(2,int(5),figsize=(20,10))
    for i in range(int(10)):
        if i<=4:
            ax2[0][i].plot(y, y_test[i,:], label="True Volterra path")
            ax2[0][i].plot(y, model.predict((np.expand_dims(X_test[0][i,:],0),X_test[1])).squeeze(0), label="Estimated Volterra path")
        else:
            ax2[1][i-5].plot(y, y_test[i,:], label="True Volterra path")
            ax2[1][i-5].plot(y, model.predict((np.expand_dims(X_test[0][i,:],0),X_test[1])).squeeze(0), label="Estimated Volterra path")
    plt.suptitle(r'10 paths from the test set: n={}, T={}, dt={}'.format(n,T,dt))
    plt.legend()
    plt.savefig(os.path.join(foldername, "test_paths.png"))
    plt.clf()