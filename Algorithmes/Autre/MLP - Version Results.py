from numpy import linspace, random, mean, std, maximum
from numpy import ones, outer, array, diag, dot, vectorize, eye, zeros, tensordot
from numpy import exp,cos, sqrt,log, sin

from matplotlib.pyplot import plot, scatter, axes, xlabel, ylabel, clf, legend, subplot, savefig, title, fill_between
from sys import stdout

from itertools import product
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


""" --------------------------------------------- """

#https://apprendre-le-deep-learning.com/coder-reseau-de-neurones-from-scratch/

class Affine:
  def __init__(self, n, m,sigma):
    self.n = n                                                        # input dimension
    self.m = m                                                        # output dimension
    self.A = random.normal(0, 1, (n, m))                              # Weight
    self.b = random.normal(0,1,(m,))                                  # Bias
    self.D_A = zeros((n,m))
    self.D_b = zeros((m,))

  def eval(self, X):
    Y = X@self.A + self.b                                             # Linear transformation
    return Y
       
  def backpropagation(self, X, D_Y):
    self.D_b += D_Y                                                   # gradient bias
    self.D_A += tensordot(X, D_Y, axes=0)                             # gradient weight
 
    D_X = self.A@D_Y  # gradient X
    return D_X

  def update(self, alpha):
    self.A -= alpha * self.D_A                                        # Update weight
    self.b -= alpha * self.D_b                                        # Update bias

    self.D_A = zeros((self.n,self.m))                                 # reset gradient
    self.D_b = zeros(shape=(self.m,))                                 # reset gradient


class ReLU:
  def __init__(self):
    pass

  def eval(self, X):
    Y = maximum(0,X)
    return Y

  def backpropagation(self, X, D_Y):
    D_X = D_Y * (X>=0)
    return D_X

  def update(self, alpha):
    pass # Nothing to do


class Sigmoid:
  def __init__(self):
    pass

  def eval(self, X):
    Y = 1 / (1 + np.exp(-X))  # sigmoid activation
    return Y

  def backpropagation(self, X, gradient_Y):
    expo = np.exp(-X)
    gradient_X = gradient_Y * expo/square(1+expo) # gradient X
    return gradient_X

  def update(self, alpha):
    pass # Nothing to do

class MLP_Network:
  def __init__(self):
    self.layers = list()  # List of layers
    self.inputs = list()  # Memorize input

  def addLayer(self, layer):
    self.layers.append(layer)

  def __call__(self, X):
    self.inputs.clear()  # clear list of inputs
    Input = X            # get first input
    for layer in self.layers:
      self.inputs.append(Input)         # Memorize input
      Input = layer.eval(Input)  # get next input

    return Input

  def backpropagation(self, gradient_Y):
    gradient = gradient_Y  # get first gradient
    for i in reversed(range(len(self.layers))):
      X = self.inputs[i]                                      # Input of layer i
      gradient = self.layers[i].backpropagation(X, gradient)  # get next gradient

  def update(self, alpha):
    for layer in self.layers:
      layer.update(alpha)  # Update all layers

  def fit(self, X_train, Y_train, epochs, alpha):
    for epoch in range(epochs):    
      for i in range(len(X_train)):
        score_gradient = self(X_train[i])-Y_train[i]   # gradient of MSE
        self.backpropagation(score_gradient)           # Compute gradients
        
      self.update(alpha)                               # udpate weights



""" --------------------------------------------- """

def dataset_initialisation(y,C,M,ratio_data,Random_seed):
    random.seed(Random_seed)
    D = len(C)
    U = random.uniform(0,1,size = (M,D))
    X = ones((M,D))@diag([C[d][0] for d in range(D)])+U@diag([C[d][1]-C[d][0] for d in range(D)])
    Y = list(map(y,X))
    
    X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=ratio_data, random_state=42)
    return X_train, X_test, Y_train, Y_test

def MLP_regression(P_min, P_max, Data, Data_global, N, D,Epochs,alpha):
    X_train, X_test, Y_train, Y_test = Data
    X_global, Y_global = Data_global
    Train_error, Test_error, Global_error  = [], [], []
    
    for P in range(P_min, P_max+1):
        stdout.write("P: "+ str(P) + " P/N: "+ str(P/N) + "\n")

        Sigma = 1
        MLP = MLP_Network()
        MLP.addLayer( Affine(D, P,Sigma) )
        MLP.addLayer( ReLU() )
        MLP.addLayer( Affine(P, 1,Sigma ))
        
        MLP.fit(X_train, Y_train, epochs=Epochs, alpha=alpha)

        try:
          Train_error.append(log(1+mean_squared_error(Y_train, list(map(MLP,X_train)))))
        except:
          Train_error.append(Train_error[-1])

        try:
          Test_error.append(log(1+mean_squared_error(Y_test, list(map(MLP,X_test)))))
        except:
          Test_error.append(Test_error[-1])

        try:
          Global_error.append(log(1+mean_squared_error(Y_global, list(map(MLP,X_global)))))
        except:
          Global_error.append(Global_error[-1])

        stdout.write("Train_error: "+ str(Train_error[-1])+ "| Test_error: " + str(Test_error[-1])+"| Global_error: " + str(Global_error[-1])+"\n")
        
        if  P > N and N-P_min > 0:
            stdout.write("Under-parameterised minimum test: "+ str(min(Test_error[:N-P_min])) + "| Over-parameterised minimum test: " + str(min(Test_error[N-P_min:]))+"\n")
            
    
    return Train_error, Test_error, Global_error, MLP
    
def main(n, M, D, C, y, ratio_data, P_min, P_max, Epochs, alpha, Random_seed):
    
    #Dataset creation, with seed set for reproducibility
    
    Data = dataset_initialisation(y,C,M,ratio_data, Random_seed)

    N = len(Data[0])
    stdout.write("M: " + str(M) + "| N:" + str(N)+"\n")
    
    # Uniform mesh for global error calculation and visualization, only works for cubes in dimension 2
    X_global = array(list(product(linspace(C[0][0],C[0][1],n),repeat = D)))
    Y_global = list(map(y,X_global))
    Data_global = X_global, Y_global
    
    if D == 2:
        u1 = linspace(C[0][0],C[0][1],n)
        u = outer(linspace(C[0][0], C[0][1], n), ones(n))

    #Ridge regression
    Train_error, Test_error, Global_error, y_chap = MLP_regression(P_min, P_max, Data, Data_global, N, D, Epochs, alpha)
    
    # Visualization
    clf()
    
    if D == 2:
        # 2D graph visualization
        Y_visualization = array([[y((u1[i],u1[j])) for i in range(n)] for j in range(n)])
        Y_visualization_approx = array([[y_chap((u1[i],u1[j]))[0] for i in range(n)] for j in range(n)])
        ax = axes(projection='3d')
        ax.set_title('Function plot')
        ax.plot_surface(u,u.T,Y_visualization , cmap='viridis',edgecolor='green', label = "Target")
        ax.plot_surface(u,u.T,Y_visualization_approx , cmap='viridis',edgecolor='red', label = "Estimator")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        

    if D == 1:
        # 1D graph visualization
        plot(X_global,list(map(y_chap,X_global)), label="Estimator")
        plot(X_global,Y_global, label = "Target")
        scatter(Data[0], Data[2], label = "Interpolation points")
        scatter(Data[1], Data[3], label = "Test points")
        xlabel("x")
        ylabel("y")
        title("Functions plot MLP")
        legend()
        
    savefig('Function_plot_MLP.png')
    # Display of error curves
    
    clf()

    liste_P = [P/N for P in range(P_min,P_max+1)]
    plot(liste_P, Train_error, label="Train_error")
    plot(liste_P, Test_error, label="Test_error")
    plot(liste_P,Global_error, label="Global_error")
    xlabel("P/N ratio with P:nb parameters, N: nb points")
    ylabel("log(1+MSE)")
    title("Errors plot MLP")
    legend()
    
    savefig('Graph_error_MLP.png')
    
    return Train_error, Test_error, Global_error
    
a, n_example = 1, 1
D= 2

n = 100
M = 40
r = 0.2
Epochs = 10000
alpha = 10**(-4)
P_min,P_max = 1, 25
Liste_seeds = [1000,12334,13902,29389,1984,2003,1003,2039,1001,1002]
ratio_complex = 3

if n_example == 1:
    y = lambda X: 2*exp(D - sum([X[i]**2 for i in range(D)]))
if n_example == 2:
    y = lambda X: sum([3*cos(10*X[i]) + cos(X[i])  - X[i]**2 for i in range(D)])
if n_example == 3:
    y = lambda X: sum([20*sin(3*X[i]) + sin(X[i])  - X[i]**3 for i in range(D)]) 
C = array([[-a,a] for _ in range(D)])

N = int(M*(1-r))
liste_P = [ratio_complex*P/N for P in range(P_min,P_max+1)]


Liste_Beta_norm, Liste_Train_error, Liste_Test_error, Liste_Global_error = [],[],[],[]
for Random_seed in Liste_seeds:
    Train_error, Test_error, Global_error = main(n, M, D, C, y, r, P_min, P_max, Epochs, alpha, Random_seed)
    Liste_Train_error.append(Train_error)
    Liste_Test_error.append(Test_error)
    Liste_Global_error.append(Global_error)

Train_error_min, Train_error_max = mean(Liste_Train_error,axis=0)-(1/2)*std(Liste_Train_error, axis=0), mean(Liste_Train_error,axis=0)+(1/2)*std(Liste_Train_error, axis=0)
Test_error_min, Test_error_max = mean(Liste_Test_error,axis=0)-(1/2)*std(Liste_Test_error, axis=0), mean(Liste_Test_error,axis=0)+(1/2)*std(Liste_Test_error, axis=0)
Global_error_min, Global_error_max = mean(Liste_Global_error,axis=0)-(1/2)*std(Liste_Global_error, axis=0), mean(Liste_Global_error,axis=0)+(1/2)*std(Liste_Global_error, axis=0)

clf()
fill_between(liste_P, Train_error_min, Train_error_max, color='orange', alpha=0.3)
fill_between(liste_P, Test_error_min, Test_error_max, color='green', alpha=0.3)
fill_between(liste_P, Global_error_min, Global_error_max, color='red', alpha=0.3)

plot(liste_P, mean(Liste_Train_error,axis=0), color='orange', label="Train_error_mean")
plot(liste_P, mean(Liste_Test_error,axis=0), color='green', label="Test_error_mean")
plot(liste_P,mean(Liste_Global_error,axis=0), color='red', label="Global_error_mean")

xlabel("P/N ratio with P:nb parameters, N: nb points")
ylabel("log(1+MSE)")
title("Mean errors MLP")
legend()
savefig('Graph_error_MLP_results.png')
