import torch
import torch.nn as nn
import numpy as np
import math
import os
import operator
from functools import reduce
from timeit import default_timer
import yaml
import warnings
warnings.simplefilter("ignore", UserWarning)
from datetime import datetime
import tqdm
import matplotlib.pyplot as plt

#---------------------------------------------------------------------
def read_yaml(file_path):           # To read the configurations file
    with open(file_path, "r") as f:
        return yaml.safe_load(f)
    
is_cuda = torch.cuda.is_available()     # check if gpu is availaböe
device = 'cuda' if is_cuda else 'cpu'
if not is_cuda:
    print("Warning: CUDA not available; falling back to CPU but this is likely to be very slow.")

def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c

#-------Loading the parameters-----------------------------------------
params = read_yaml("configurations.yaml") # Load the parameters
hidden_channels, hidden_states_kernels, epochs, batch_size, learning_rate, scheduler_gamma, p, n, dim, T, dt, print_every, sve_type, epsilon, theta_o, mu, sigma, alpha, kappa, theta, xi, x0 = params.values()
ntrain=int(n*0.8)   # Derived parameters
ntest=int(n*0.2)
scheduler_step=int(epochs/4)+1     # after every 25% of Epochs, we want to decrease the step size by scheduler_gamma multiplicatively
#---------Foldername-----------------------------------
absolute_path = os.getcwd() 
foldername = os.path.join(absolute_path, "nSVE_{}_T{}_dt{}_hidden{}_epochs{}_n{}".format(sve_type,T, dt, hidden_channels, epochs, n))  # where to store the plots
if not os.path.isdir(foldername): os.makedirs(foldername)
else:   # if foldername already exists, we additionally take thetime to the foldername to avoid duplicates
    now = str(datetime.now()).replace(" ", "-").replace(":",".") 
    foldername = os.path.join(absolute_path, "nSVE_{}_T{}_dt{}_hidden{}_epochs{}_n{}_time{}".format(sve_type,T, dt, hidden_channels, epochs, n, now))
    os.makedirs(foldername)

#-----------Solving a SVE----------------------------
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
    # Create  dim-dimensional Brownian motion with time step = dt
    def BM(self, start, stop, dt, l, dim):
        T = self.partition(start, stop, dt)
        # assign to each point of len(T) time point an N(0, \sqrt(dt)) standard l dimensional random variable
        BM = np.random.normal(scale=np.sqrt(dt), size=(len(T), l, dim))
        BM[0] = 0 #set the initial value to 0
        BM = np.cumsum(BM, axis  = 0) # cumulative sum: B_n = \sum_1^n N(0, \sqrt(dt))
        return BM

X_0 = np.random.normal(x0,x0/10,[n, dim])  # Initial value
B = Noise().BM(0, T, dt, n, dim)     # Brownian motion realizations

# Generate SVE parameters dependent on sve_type
if sve_type == "PEN":   # Pendulum equation
    g = lambda t: 1
    K_mu = lambda s, t: s-t
    K_sigma = lambda s, t: epsilon*(s-t)
    mu = lambda s, x: x
    sigma = lambda s,x: x
elif sve_type == "OU":   # Ornstein-Uhlenbeck     
    g = lambda t: np.exp(-theta_o*t)
    K_mu = lambda s, t: np.exp(-theta_o*(t-s))
    K_sigma = lambda s, t: np.exp(-theta_o*(t-s))
    mu = lambda s, x: x
    sigma = lambda s,x: np.sqrt(np.abs(x))
elif sve_type == "RH":   # Rough-Heston
    g = lambda t: 1
    K_mu = lambda s, t: np.power((t-s), -alpha)
    K_sigma = lambda s, t: np.power((t-s), -alpha)
    mu = lambda s, x: kappa*(theta-x)/math.gamma(1-alpha)
    sigma = lambda s,x: xi * np.sqrt(np.abs(x))/math.gamma(1-alpha)
elif sve_type == "JUMP":   # SVE with kernels K(t-s)=1 if t-s<=T/4 and K(t-s)=-1 else
    g = lambda t: 1
    K_mu = lambda s, t: -1+2*np.sign(T/4-(t-s))
    K_sigma = lambda s, t: -1+2*np.sign(T/4-(t-s))
    mu = lambda s, x: theta-x
    sigma = lambda s,x: np.sqrt(np.abs(x))

Sol = SVE(g, K_mu, K_sigma, mu, sigma).solver_(B, X_0, T, dt)   
#---------------------------------------------------

B_t = torch.from_numpy(B.astype(np.float32)).to(device)  # tensors in pytorch
data = torch.from_numpy(Sol.astype(np.float32)).to(device)

def dataloader_SVE(data, B, ntrain, ntest, batch_size):
    X_0_train = data[0,:ntrain,:].unsqueeze(0).transpose(0,1) # switch dimensions to get same in the first one. Now: (n, t, dim)
    X_train = data[:,:ntrain,:].transpose(0,1) # switch dimensions
    B_train = B[:,:ntrain,:].transpose(0,1) # switch dimensions

    X_0_test = data[0,-ntest:,:].unsqueeze(0).transpose(0,1) # switch dimensions
    X_test = data[:,-ntest:,:].transpose(0,1) # switch dimensions
    B_test = B[:,-ntest:,:].transpose(0,1) # switch dimensions

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_0_train, B_train, X_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_0_test, B_test, X_test), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

train_loader, test_loader = dataloader_SVE(data, B_t, ntrain=ntrain, ntest=ntest, batch_size=batch_size)

class solver_neuralSVE(nn.Module):
    def __init__(self, g, K_mu, K_sigma, mu, sigma):
        super().__init__()
        self.g = g
        self.K_mu = K_mu
        self.K_sigma = K_sigma
        self.mu = mu
        self.sigma = sigma
    
    def forward(self, B, z_0, T, dt, hidden_channels, dim, batch_size):  # solver of a SVE in the latent space
        # assume B, X_0,T,dt to be torch.tensors
        num =int(T/dt)   # number of grid points
        if T/dt > num: raise ValueError("dt is not of the form 1/m")
        T = torch.tensor(T, device=B.device)
        dt = torch.tensor(dt, device=B.device)
        Solution = torch.zeros(B.size(1), B.size(0), z_0.size(2), device=B.device) # correct shape
        if len(Solution.shape) != 3: raise ValueError("B not in 3d (time, n, dim)")
        o = torch.tensor([0], device=B.device).float()
        o2 = torch.zeros([1,z_0.shape[0]], device=B.device)
        Solution[0, :, :] = z_0.squeeze(1)*self.g(o) # z_0 is given
        Solution[1,:,:] = z_0.squeeze(1)*self.g((dt).unsqueeze(0)) + self.K_mu(dt-o) * self.mu(torch.cat((o2, z_0.squeeze(1).transpose(0,1))).transpose(0,1)) * dt + (self.K_sigma(dt-o) * torch.bmm(self.sigma(torch.cat((o2, z_0.squeeze(1).transpose(0,1))).transpose(0,1)).view(batch_size, hidden_channels, dim), B[:,1,:].unsqueeze(2))).squeeze(2)  # first step in the E.-M.-scheme
        dt_extended = dt*torch.ones_like(z_0[:,:,0], device=B.device).T
        if z_0.shape[0] == 1: dt_extended = dt_extended.squeeze(1)
        # E.-M.-scheme
        for i in range(1, num):
            int_val = sum([self.K_mu((i+1-j)*dt.unsqueeze(0)) * self.mu(torch.cat((j*dt_extended, Solution[j].squeeze(0).T)).T) * dt + (self.K_sigma((i+1-j)*dt.unsqueeze(0)) * torch.bmm(self.sigma(torch.cat((j*dt_extended, Solution[j].squeeze(0).T)).T).view(batch_size, hidden_channels, dim), (B[:,j+1,:] - B[:,j,:]).unsqueeze(2))).squeeze(2) for j in range(1,i+1)])
            Solution[i+1,:,:] = z_0.squeeze(1)*self.g(((i+1)*dt).unsqueeze(0)) + int_val + self.K_mu(((i+1)*dt).unsqueeze(0)) * self.mu(torch.cat((o2, z_0.squeeze(1).transpose(0,1))).transpose(0,1))*dt + (self.K_sigma(((i+1)*dt - o)) * torch.bmm(self.sigma(torch.cat((o2, z_0.squeeze(1).transpose(0,1))).transpose(0,1)).view(batch_size, hidden_channels, dim), B[:,1,:].unsqueeze(2))).squeeze(2)
        return Solution

class LipSwish(torch.nn.Module): # we use LipSwish activations in the entire model
    def forward(self, x):
        return 0.909 * torch.nn.functional.silu(x)

class SVE_NN(torch.nn.Module):
    def __init__(self, noise_channels, hidden_channels, hidden_states_kernels, dim):
        super().__init__()
        self.noise_channels = noise_channels
        self.hidden_channels = hidden_channels

        model_g = [nn.Linear(1, hidden_states_kernels), LipSwish(), nn.Linear(hidden_states_kernels, hidden_states_kernels), LipSwish(), nn.Linear(hidden_states_kernels, 1), LipSwish()]
        self.g = nn.Sequential(*model_g)
        model_mu = [nn.Linear(1+hidden_channels, hidden_channels), LipSwish(), nn.Linear(hidden_channels, hidden_channels)]#[nn.Conv1d(1 + hidden_channels, hidden_channels, 1), nn.BatchNorm1d(hidden_channels), nn.Tanh()]
        self.mu = nn.Sequential(*model_mu)
        model_sigma = [nn.Linear(1+hidden_channels, hidden_channels*dim), LipSwish(), nn.Linear(hidden_channels*dim, hidden_channels*dim)]#[nn.Conv1d(1 + hidden_channels, hidden_channels, 1), nn.BatchNorm1d(hidden_channels), nn.Tanh()]
        self.sigma = nn.Sequential(*model_sigma)
        model_K_mu = [nn.Linear(1, hidden_states_kernels), LipSwish(), nn.Linear(hidden_states_kernels, hidden_states_kernels), LipSwish(), nn.Linear(hidden_states_kernels, 1), LipSwish()]
        self.K_mu = nn.Sequential(*model_K_mu)
        model_K_sigma = [nn.Linear(1, hidden_states_kernels), LipSwish(), nn.Linear(hidden_states_kernels, hidden_states_kernels), LipSwish(), nn.Linear(hidden_states_kernels, 1), LipSwish()]
        self.K_sigma = nn.Sequential(*model_K_sigma)

class NeuralSVE(torch.nn.Module):
    def __init__(self, dim, noise_channels, hidden_channels, hidden_states_kernels, batch_size, T, dt):
        super().__init__()
        """
        dim: dimension of the model
        noise_channels: dimension of the noise: Brownian motion
        hidden_channels: dimension of the latent space
        """
        self.dim = dim
        self.hidden_channels = hidden_channels
        self.batch_size = batch_size
        self.T = T
        self.dt = dt
        self.lift = nn.Linear(dim, hidden_channels) # initial lift
        self.SVE = SVE_NN(noise_channels, hidden_channels, hidden_states_kernels, dim)
        self.solver = solver_neuralSVE(self.SVE.g, self.SVE.K_mu, self.SVE.K_sigma, self.SVE.mu, self.SVE.sigma)
        readout = [nn.Linear(hidden_channels, dim)] # readout
        self.readout = nn.Sequential(*readout)
        
    def forward(self, x0, B):
        # solve the SVE
        z0 = self.lift(x0)   
        z = self.solver(B, z0, self.T, self.dt, self.hidden_channels, self.dim, batch_size=z0.shape[0])  # Solve SVE in latent space
        x = self.readout(z)
        return x


model = NeuralSVE(dim, noise_channels=dim, hidden_channels=hidden_channels, hidden_states_kernels=hidden_states_kernels, batch_size=batch_size, T=T, dt=dt).to(device)
absolute_path = os.getcwd()
if not os.path.isdir(os.path.join(absolute_path,"Models")): os.makedirs(os.path.join(absolute_path,"Models"))  # to load/store the model
relative_path2 = os.path.join("Models", "{}_hidden{}_hsk{}.pth".format(sve_type,hidden_channels,hidden_states_kernels))
full_path2 = os.path.join(absolute_path, relative_path2)
try: 
    model.load_state_dict(torch.load(full_path2))
    print("Model loaded.")
except:
    print("No model found yet.")
print('The model has {} parameters'. format(count_params(model)))
model = model.float()

class LpLoss(object):
    def __init__(self, dim=1, p=2):
        super(LpLoss, self).__init__()
        self.dim = dim
        self.p = p
    def rel(self, x, y):
        diff_norms = torch.norm(x - y.transpose(0,1), self.p, 0)
        y_norms = torch.norm(y.transpose(0,1), self.p, 0)
        return torch.sum(diff_norms/y_norms)
    
loss =LpLoss(p=p)  

# Start the training
def train_nsve(model, train_loader, test_loader, myloss, epochs=5000, learning_rate=0.001, scheduler_step=100, scheduler_gamma=0.5, print_every=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma) # alternativen zur adaptiven Stepsize Steuerung möglich

    ntrain = len(train_loader.dataset)
    ntest = len(test_loader.dataset)

    # to store for output:
    losses_train = [] 
    losses_test = []
    times_train = [] 
    times_eval = []
    output_train = []
    output_model_train = []
    output_test = []
    output_model_test = []
    try:
        trange = tqdm.tqdm(range(epochs))
        for ep in trange:
            model.train()       
            train_loss = 0.

            for X0_, B_, X_ in train_loader:
                loss = 0.
                t1 = default_timer()
                x_pred = model(X0_, B_)
                if ep==epochs-1:    # store at the end of training
                    output_train.append(X_)
                    output_model_train.append(x_pred.transpose(0,1))
                loss = myloss.rel(x_pred, X_)

                train_loss += loss.item()
                loss.backward()  # backward calculation of partial derivatives
                optimizer.step()
                optimizer.zero_grad()

                times_train.append(default_timer()-t1)

            # testing on test set
            test_loss = 0.
            with torch.no_grad(): # no gradient calculations on test set
                for X0_, B_, X_ in test_loader:
                    loss = 0.
                    t1 = default_timer()
                    x_pred = model(X0_, B_)
                    if ep==epochs-1:
                        output_test.append(X_)
                        output_model_test.append(x_pred.transpose(0,1))
                    loss = myloss.rel(x_pred, X_)
                    test_loss += loss.item()
                    times_eval.append(default_timer()-t1)
            scheduler.step()

            if ep % print_every == 0:
                losses_train.append(train_loss/ntrain)
                losses_test.append(test_loss/ntest)
                trange.write("Epoch {:04d} | Total Train Loss {:.6f} | Total Test Loss {:.6f}".format(ep, train_loss / ntrain, test_loss / ntest))
        return model, losses_train, losses_test, output_train, output_model_train, output_test, output_model_test

    except KeyboardInterrupt:
        return model, losses_train, losses_test, output_train, output_model_train, output_test, output_model_test
    

model, losses_train, losses_test, output_train, output_model_train, output_test, output_model_test = train_nsve(model, train_loader, test_loader, loss, epochs=epochs, learning_rate=learning_rate, scheduler_step=scheduler_step, scheduler_gamma=scheduler_gamma, print_every=print_every)   # epochs bei n=1200 war 5000

torch.save(model.state_dict(), full_path2)

# Filenamesfor the plots:
relative_path3_l = os.path.join(foldername, "{}_T{}_dt{}_hidden{}_hsk{}_loss.png".format(sve_type, T,dt,hidden_channels, hidden_states_kernels))
full_path3_l = os.path.join(absolute_path, relative_path3_l)
relative_path3_p1 = os.path.join(foldername, "{}_T{}_dt{}_hidden{}_hsk{}_pathstrain.png".format(sve_type, T,dt,hidden_channels, hidden_states_kernels))
full_path3_p1 = os.path.join(absolute_path, relative_path3_p1)
relative_path3_p2 = os.path.join(foldername, "{}_T{}_dt{}_hidden{}_hsk{}_pathstest.png".format(sve_type, T,dt,hidden_channels, hidden_states_kernels))
full_path3_p2 = os.path.join(absolute_path, relative_path3_p2)

if n<50: print("Only generating plots for n>=50.")
if dim==1 and n>=50:
    plt.plot(np.arange(1,len(losses_train)*print_every+1, print_every), losses_train, label='train')
    plt.plot(np.arange(1,len(losses_test)*print_every+1, print_every), losses_test, label='test')
    plt.xlabel('Epoch')
    plt.ylabel('Relative L2 loss')
    plt.legend()
    plt.savefig(full_path3_l)
    plt.clf()

    if T<1: y = T * np.linspace(0,1,int(T/dt+1))
    else: y = np.linspace(0,int(T),int(T/dt+1))
    fig, ax = plt.subplots(2,int(5),figsize=(20,10))
    for i in range(int(10)):
        if i<=4:
            ax[0][i].plot(y, output_train[0][i].cpu(), label="True Volterra path")
            ax[0][i].plot(y, output_model_train[0][i].cpu().detach().numpy(), label="Estimated Volterra path")
        else:
            ax[1][i-5].plot(y, output_train[0][i].cpu(), label="True Volterra path")
            ax[1][i-5].plot(y, output_model_train[0][i].cpu().detach().numpy(), label="Estimated Volterra path")
    plt.suptitle(r'10 paths from the training set: n={}, T={}, dt={}, hidden_channels={}'.format(n,T,dt,hidden_channels))
    plt.legend()
    plt.savefig(full_path3_p1)
    plt.clf()

    fig2, ax2 = plt.subplots(2,int(5),figsize=(20,10))
    for i in range(int(10)):
        if i<=4:
            ax2[0][i].plot(y, output_test[0][i].cpu(), label="True Volterra path")
            ax2[0][i].plot(y, output_model_test[0][i].cpu().detach().numpy(), label="Estimated Volterra path")
        else:
            ax2[1][i-5].plot(y, output_test[0][i].cpu(), label="True Volterra path")
            ax2[1][i-5].plot(y, output_model_test[0][i].cpu().detach().numpy(), label="Estimated Volterra path")
    plt.suptitle(r'10 paths from the testing set: n={}, T={}, dt={}, hidden_channels={}'.format(n,T,dt,hidden_channels))
    plt.legend()
    plt.savefig(full_path3_p2)
    plt.clf()

if dim==2 and n>=50:
    fig, ax = plt.subplots(2,int(5),figsize=(20,10))
    for i in range(int(10)):
        x = output_train[0][i,:,0]
        y = output_train[0][i,:,1]
        x_model = output_model_train[0][i,:,0].cpu().detach().numpy()
        y_model = output_model_train[0][i,:,1].cpu().detach().numpy()
        if i<=4:
            ax[0][i].plot(x,y, label="True Volterra path", c="b")
            ax[0][i].plot(x_model,y_model, label="Estimated Volterra path", c="r")
        else:
            ax[1][i-5].plot(x,y, label="True Volterra path", c="b")
            ax[1][i-5].plot(x_model,y_model, label="Estimated Volterra path", c="r")
    plt.suptitle(r'10 paths from the training set: n={}, T={}, dt={}, hidden_channels={}'.format(n,T,dt,hidden_channels))
    plt.legend()
    plt.savefig(full_path3_p1)
    plt.clf()

    fig2, ax2 = plt.subplots(2,int(5),figsize=(20,10))
    for i in range(int(10)):
        x = output_test[0][i,:,0]
        y = output_test[0][i,:,1]
        x_model = output_model_test[0][i,:,0].cpu().detach().numpy()
        y_model = output_model_test[0][i,:,1].cpu().detach().numpy()
        if i<=4:
            ax2[0][i].plot(x,y, label="True Volterra path", c="b")
            ax2[0][i].plot(x_model,y_model, label="Estimated Volterra path", c="r")
        else:
            ax2[1][i-5].plot(x,y, label="True Volterra path", c="b")
            ax2[1][i-5].plot(x_model,y_model, label="Estimated Volterra path", c="r")
    plt.suptitle(r'10 paths from the test set: n={}, T={}, dt={}, hidden_channels={}'.format(n,T,dt,hidden_channels))
    plt.legend()
    plt.savefig(full_path3_p2)
    plt.clf()

# Last: store the configs
with open(os.path.join(foldername,'configs.txt'), 'w') as f:
    for p in params:
        f.write("{}: {}\n".format(p,params[p]))