# Four layer neural network from scratch to predict the credit default.
# It contains train as well as test model  

# python library import
import numpy as np
import pandas as pd
import time
import sys

no_of_iteration = int(sys.argv[1])

# starting time of the program
start_time = time.clock()
print start_time, "seconds"

# sigmoid function
def sigmoid(x):
	return 1/(1+np.exp(-x))
	# return np.tanh(x)
# sigmoidPrime function - Here we will pass output of sigmoid function
def sigmoidPrime(x):
	return x*(1-x)
	# return 1.0 - x**2

# parameter initialization 

# alphas is a array which can be used to run the code for multiple values of alpha in one go.
#alphas = [0.001,0.01,0.1,1,10,100]
alphas = [100]

# No of neurons per layer, the first layer consists the more neuron than the input layer
inputLayer_size = 83
first_hiddenLayer_size = 166
second_hiddenLayer_size = 83
third_hiddenLayer_size = 10
outputLayer_size = 2

# reading the train data from file
train_df = pd.read_csv('train.csv')

# moving the read data from frame to numpy array X
XFull = np.array(train_df.ix[0:, 0:23])

# reading the class from train data set into numpy array Y
YFull = np.array(train_df.ix[0:, -1:])

X1 = XFull[0:20000]
YTrain = YFull[0:20000]

XTest = XFull[20000:30000]
YTest = YFull[20000:30000]

# calculating the no of train data records
no_of_train_data_row=len(X1)

X=np.zeros((no_of_train_data_row,83))

for i in range(0,no_of_train_data_row):
	for j in range(24):
		if(j==0):
			X[i][0]=X1[i][j]
		if(j==1):
			index=X1[i][j]
			X[i][1+index-1]=1
		if(j==2):
			index=X1[i][j]
			X[i][3+index-1]=1
		if(j==3):
			index=X1[i][j]
			X[i][7+index-1]=1
		if(j==4):
			X[i][10]=X1[i][j]
		if(j==5):
			index=X1[i][j]
			if(index==-1):
				X[i][13+index-1]=1
			else:
				X[i][12+index-1]=1
		if(j==6):
			index=X1[i][j]
			if(index==-1):
				X[i][23+index-1]=1
			else:
				X[i][22+index-1]=1
		if(j==7):
			index=X1[i][j]
			if(index==-1):
				X[i][33+index-1]=1
			else:
				X[i][32+index-1]=1
		if(j==8):
			index=X1[i][j]
			if(index==-1):
				X[i][43+index-1]=1
			else:
				X[i][42+index-1]=1
		if(j==9):
			index=X1[i][j]
			if(index==-1):
				X[i][53+index-1]=1
			else:
				X[i][52+index-1]=1
		if(j==10):
			index=X1[i][j]
			if(index==-1):
				X[i][63+index-1]=1
			else:
				X[i][62+index-1]=1
		if(j==11):
			X[i][71]=X1[i][j]
		if(j==12):
			X[i][72]=X1[i][j]
		if(j==13):
			X[i][73]=X1[i][j]
		if(j==14):
			X[i][74]=X1[i][j]
		if(j==15):
			X[i][75]=X1[i][j]
		if(j==16):
			X[i][76]=X1[i][j]
		if(j==17):
			X[i][77]=X1[i][j]
		if(j==18):
			X[i][78]=X1[i][j]
		if(j==19):
			X[i][79]=X1[i][j]
		if(j==20):
			X[i][80]=X1[i][j]
		if(j==21):
			X[i][81]=X1[i][j]
		if(j==22):
			X[i][82]=X1[i][j]

# TRAINGING ALGO
# loop to run the code through various learning rate in on go
for alpha in alphas:
	print "Training With Alpha:" + str(alpha)
	np.random.seed(1)

	# randomly initialize our weights with mean 0
	W0 = 2*np.random.random((inputLayer_size,first_hiddenLayer_size)) - 1
	W1 = 2*np.random.random((first_hiddenLayer_size,second_hiddenLayer_size)) - 1
	W2 = 2*np.random.random((second_hiddenLayer_size,third_hiddenLayer_size)) - 1
	W3 = 2*np.random.random((third_hiddenLayer_size,outputLayer_size)) - 1

	# dictionary data structure, which will be used to store the model 
	model = {}

	# loop to run the code for multiple iteration
	for i in xrange(0, no_of_iteration):
		# variable initialization for one iteration
		k=0
		error=0 
		batch_size=100 # It stores the batch size, which we process in single go
		match_count=0

		# numpy array to store the calculated value of class after each iteration, it will be used to calculate the no of match and difference  
		Y_calculated=np.empty((no_of_train_data_row,1))

		# loop to iterate over all training data in the size of batch of 100 records 
		for j in range(0,no_of_train_data_row/batch_size):
			# forward propagation for layer 0,1,2,3 and 4

			# taking the train data feature into l0 in batch size
			l0 = X[k:k+batch_size]

			# taking the dot production of input at each layer with the weight and applying the sigmoid on that
			l1 = sigmoid(np.dot(l0,W0))
			l2 = sigmoid(np.dot(l1,W1))
			l3 = sigmoid(np.dot(l2,W2))
			l4 = sigmoid(np.dot(l3,W3))

			# taking the train data class into y_batch in batch size
			y_batch = YTrain[k:k+batch_size]

			# difference of actual value and calculated value, it will be using the back propagation, 
			# it will also we used to calculate the error
			l4_error = y_batch - l4

			#storing the predicted data into Y_calculated array after converting it into corresponding class
			for l in range(0,batch_size):
				Y_calculated[k+l]=np.argmax(l4[l])

				# calculating the no of matched records
				if(Y_calculated[k+l]==YTrain[k+l]):
					match_count=match_count+1

			# error calculations
			if(j==0):
				error=np.mean(np.abs(l4_error))
			else:
				error=(error+np.mean(np.abs(l4_error)))/2

			# back propagation using gradient descent, the calculation is same for each layer
			# delta will be calculated by multipling the error and sigmoidPrime of calculated value at each layer
			l4_delta = l4_error*sigmoidPrime(l4)

			# error will be calculated by multipling the delta at next layer and weight at that layer
			l3_error = l4_delta.dot(W3.T)
			l3_delta = l3_error*sigmoidPrime(l3)
			l2_error = l3_delta.dot(W2.T)
			l2_delta = l2_error*sigmoidPrime(l2)
			l1_error = l2_delta.dot(W1.T)
			l1_delta = l1_error*sigmoidPrime(l1)

			# updating the weight matrix by adding the alpha times output at that layer and then dot product with delta value at next layer
			W3 += alpha * l3.T.dot(l4_delta)
			W2 += alpha * l2.T.dot(l3_delta)
			W1 += alpha * l1.T.dot(l2_delta)
			W0 += alpha * l0.T.dot(l1_delta)

			# store the calculated weight into the dictionary model
			model = { 'W0': W0, 'W1': W1,'W2': W2,'W3': W3}

			# increase the k by batch size
			k=k+batch_size

		# printing the error, match count and difference after each two interation
		if (i% 10) == 0:
			print "Error:  " + str(error)
			print "match_count:  " + str(match_count)
			print "difference:  " + str(no_of_train_data_row-match_count)


# reading test data from file to get the prediction 
# test_df = pd.read_csv('test.csv')
# storing test data from frame to numpy array 
# X_temp1 = np.array(test_df.ix[0:, 1:24])
X1 = XTest
# calculating the length of test data
no_of_test_data_row=len(XTest)

X_test=np.zeros((no_of_test_data_row,83))
X = X_test

for i in range(0,no_of_test_data_row):
	for j in range(24):
		if(j==0):
			X[i][0]=X1[i][j]
		if(j==1):
			index=X1[i][j]
			X[i][1+index-1]=1
		if(j==2):
			index=X1[i][j]
			X[i][3+index-1]=1
		if(j==3):
			index=X1[i][j]
			X[i][7+index-1]=1
		if(j==4):
			X[i][10]=X1[i][j]
		if(j==5):
			index=X1[i][j]
			if(index==-1):
				X[i][13+index-1]=1
			else:
				X[i][12+index-1]=1
		if(j==6):
			index=X1[i][j]
			if(index==-1):
				X[i][23+index-1]=1
			else:
				X[i][22+index-1]=1
		if(j==7):
			index=X1[i][j]
			if(index==-1):
				X[i][33+index-1]=1
			else:
				X[i][32+index-1]=1
		if(j==8):
			index=X1[i][j]
			if(index==-1):
				X[i][43+index-1]=1
			else:
				X[i][42+index-1]=1
		if(j==9):
			index=X1[i][j]
			if(index==-1):
				X[i][53+index-1]=1
			else:
				X[i][52+index-1]=1
		if(j==10):
			index=X1[i][j]
			if(index==-1):
				X[i][63+index-1]=1
			else:
				X[i][62+index-1]=1
		if(j==11):
			X[i][71]=X1[i][j]
		if(j==12):
			X[i][72]=X1[i][j]
		if(j==13):
			X[i][73]=X1[i][j]
		if(j==14):
			X[i][74]=X1[i][j]
		if(j==15):
			X[i][75]=X1[i][j]
		if(j==16):
			X[i][76]=X1[i][j]
		if(j==17):
			X[i][77]=X1[i][j]
		if(j==18):
			X[i][78]=X1[i][j]
		if(j==19):
			X[i][79]=X1[i][j]
		if(j==20):
			X[i][80]=X1[i][j]
		if(j==21):
			X[i][81]=X1[i][j]
		if(j==22):
			X[i][82]=X1[i][j]

# predicting the poker class using forward propagation using trained model
l1 = sigmoid(np.dot(X,W0))
l2 = sigmoid(np.dot(l1,W1))
l3 = sigmoid(np.dot(l2,W2))
l4 = sigmoid(np.dot(l3,W3))

# storing the last layer output in y_predicted
y_predicted = l4

# numpy array to store the predicted class for poker
y_final=np.empty((no_of_test_data_row,1))

# storing the predicted class into y_final array after converting it into corresponding class
for i in range(0,no_of_test_data_row):
	y_final[i]=np.argmax(y_predicted[i])

# writing predicted data to output file
# test_df['CLASS'] = y_final.astype(int)
# test_df["id"] = test_df['CLASS'].index
# test_df[["id","CLASS"]].to_csv("output_nn.csv",index=False)


test_df = XTest
Y_Actual = YTest

count=0
for i in range(0,no_of_test_data_row):
	if(Y_Actual[i]==y_final[i]):
		count=count+1

correct_prediction=count
difference=(no_of_test_data_row-correct_prediction)  

print("correct prediction : ",correct_prediction)
print("difference : ",difference)

percentage=(correct_prediction*100)/no_of_test_data_row

print("percentage accuracy : ",percentage)

# printing the time taking to run the program
print time.clock() - start_time, "seconds"

