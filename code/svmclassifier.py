from sklearn import svm
import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.decomposition import PCA


total_df = pd.read_csv('train.csv')
XFull = np.array(total_df.ix[0:, 0:23])
YFull = np.array(total_df.ix[0:, -1:])

XTrain = XFull[0:20000]
YTrain = YFull[0:20000]

XTest = XFull[20000:30000]
YTest = YFull[20000:30000]


pca = PCA(n_components=23)

pca.fit(XTrain)
X_t_train = pca.transform(XTrain)
X_t_test = pca.transform(XTest)


# print XTest
# print YTest

clf = svm.SVC()
clf.fit(X_t_train, YTrain.ravel())


# print clf.get_params()

Y_Predicted=np.array(clf.predict(X_t_test))


count=0
for i in range(0,len(XTest)):
    if(Y_Predicted[i]==YTest[i]):
        count=count+1

correct_prediction=count
difference=(len(XTest)-correct_prediction)  

print ("SVC with rbf kernel")
print("correct prediction : ",correct_prediction)
print("difference : ",difference)

percentage=(correct_prediction*100)/len(XTest)

print("percentage accuracy : ",percentage)







clf = svm.SVC(kernel='sigmoid')
clf.fit(X_t_train, YTrain.ravel())


# print clf.get_params()

Y_Predicted=np.array(clf.predict(X_t_test))


count=0
for i in range(0,len(XTest)):
    if(Y_Predicted[i]==YTest[i]):
        count=count+1

correct_prediction=count
difference=(len(XTest)-correct_prediction)  

print ("SVC with sigmoid kernel")
print("correct prediction : ",correct_prediction)
print("difference : ",difference)

percentage=(correct_prediction*100)/len(XTest)

print("percentage accuracy : ",percentage)









clf = svm.LinearSVC()
clf.fit(X_t_train, YTrain.ravel())


# print clf.get_params()

Y_Predicted=np.array(clf.predict(X_t_test))


count=0
for i in range(0,len(XTest)):
    if(Y_Predicted[i]==YTest[i]):
        count=count+1

correct_prediction=count
difference=(len(XTest)-correct_prediction)  

print ("SVC with linear kernel, number of iterations = 1000")
print("correct prediction : ",correct_prediction)
print("difference : ",difference)

percentage=(correct_prediction*100)/len(XTest)

print("percentage accuracy : ",percentage)