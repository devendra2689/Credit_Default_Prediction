from sklearn import svm
import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

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


numberOfNeighbours = [1,2,3,4,5,6,7]
algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
leafSizes = [10,20,30,40,50]

# print XTest
print YTest

for n in numberOfNeighbours:
    for a in algorithms:
        for l in leafSizes:
            print ("n_neighbors = ",n," algorithm = ",a," leaf_size = ",l)
            clf = KNeighborsClassifier(n_neighbors=n,algorithm=a,leaf_size=l)
            clf.fit(X_t_train, YTrain.ravel())


            # print clf.get_params()

            Y_Predicted=np.array(clf.predict(X_t_test))


            count=0
            for i in range(0,len(XTest)):
                if(Y_Predicted[i]==YTest[i]):
                    count=count+1

            correct_prediction=count
            difference=(len(XTest)-correct_prediction)  

            print("correct prediction : ",correct_prediction)
            print("difference : ",difference)

            percentage=(correct_prediction*100)/len(XTest)

            print("percentage accuracy : ",percentage)





# clf = KNeighborsClassifier(n_neighbors=4,)
# clf.fit(X_t_train, YTrain.ravel())


# print clf.get_params()

# Y_Predicted=np.array(clf.predict(X_t_test))


# count=0
# for i in range(0,len(XTest)):
#     if(Y_Predicted[i]==YTest[i]):
#         count=count+1

# correct_prediction=count
# difference=(len(XTest)-correct_prediction)  

# print("correct prediction : ",correct_prediction)
# print("difference : ",difference)

# percentage=(correct_prediction*100)/len(XTest)

# print("percentage accuracy : ",percentage)