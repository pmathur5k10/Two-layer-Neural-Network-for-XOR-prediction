



####   FINDING XOR WITH 1 USING ARTIFICIAL NEURAL NETWORK OF 2 LAYERS     ####






import numpy as np
#numpy is the scientific library in python
import time
#time is the inbuilt library in python to show the time functions

#the number of neurons in each layer
n_hidden=10
n_input=10
n_output=10
# no of samples to be trained
n_samples=300

#the leaning rate::== the leaning rate determines how much the updating step influenes the local weights
#how fast the neural networks move
learning_rate=0.01
#the momentum::==momentum adds the fraction m of the previos weight update to the current one// 
#if the gradient points in the same direction, it increases the momentum
#if the gradient keeps changing the direction , the momentum decreases to smooth out the transitions
momentum=0.9

#seed a random number so as to get the same number in all training examples
#non deterministic seeding
np.random.seed(0)

#sigmoid activation function
#converts the intermediate outputs into probablity
def sigmoid(x):
	return 1/(1+np.exp(-x))


#derivative of tanh activation function tanh prime=1- sqaure of tanh
#used for better loss function for XOR problem
def tanh_prime(x):
	return 1 - np.tanh(x)**2


#input data, transpose, layer 1, layer2, biases in each layer
# layer v and w are the two layers 
# bv and bw are the biases for the two layers which make our prediction more accurate
def train(x,t,V,W,bv,bw):
#forward propogation
	A=np.dot(x,V)+bv
	Z=np.tanh(A)

	B=np.dot(Z,W)+bw
	Y=sigmoid(B)

#backward propogation

	Ew=Y-t
	Ev=tanh_prime(A)*np.dot(W,Ew)
# outer matrix multimplication is equivalent to multiplying nX1 matrix by 1Xm matrix
	dW=np.outer(Z,Ew)
	dV=np.outer(x,Ev)
#cross entropy
#better for classification based neural networks
	loss= -np.mean(t*np.log(Y)+(1-t)*np.log(1-Y))

	return loss, (dV,dW,Ev,Ew)

def predict(x,V,W,bv,bw):

	A=np.dot(x,V)+bv
	B=np.dot(np.tanh(A),W)+bw
# sigmoid B return 1 if the output is greater than 0.5 else 0
	return(sigmoid(B)>0.5).astype(int)


#np.normal.random generates random numbers from -inf to +inf in normal distribution , scale gives the difference between the numbers and 
#size gives the dimensional size of the distribution arrays
V=np.random.normal(scale=0.1,size=(n_input,n_hidden))	
W=np.random.normal(scale=0.1,size=(n_hidden,n_output))

#np.zeros gives the array of zeros of the given dimension
bv=np.zeros(n_hidden)
bw=np.zeros(n_output)

#an array containing the parameters created
param=[V,W,bv,bw]

#np.random.binomial gives a binomial distribution with n,p,size
X=np.random.binomial(1,0.5,(n_samples,n_input))

T=X^1
#loop running for 100 times each time training an example and calculating the loss function and updating it to reduce the loss at each step
for epoch in range(100):
	err=[]
	upd=[0]*len(param)
	t0=time.clock()


	for i in range(X.shape[0]):
		loss,grad=train(X[i],T[i],*param)

		
		for j in range(len(param)):


			param[j]-=upd[j]

	

		for j in range(len(param)):


			upd[j]=learning_rate*grad[j] + momentum*upd[j]

		err.append(loss)
		
	print ('Epoch: %d,Error: %.8f, Time: %.4f s' %(epoch,np.mean(err),time.clock()-t0))

#test data
q=np.random.binomial(1,0.5,(n_input))

print("PREDICTION")

print(q)
print(predict(q,*param)) 




		





