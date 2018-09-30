from sklearn import ensemble
import numpy as np

train_data = np.genfromtxt('train_data.csv', delimiter = ',')
train_data = train_data[1:].astype(int)

train_data_x = train_data[:,1:-1]
train_data_y = train_data[:, -1]

test_data = np.genfromtxt('test_data.csv', delimiter = ',')
test_data = test_data[1:].astype(int)

test_data_x = test_data[:,1:-1]
test_data_y = test_data[:, -1]

imp = ['entropy']
for imp_fn in imp:
    print(imp_fn)
    for est in range(10, 200, 10):
        decision_forest = ensemble.AdaBoostClassifier(
                                                n_estimators = est,
                                                random_state = 42
                                                )

        decision_forest.fit(train_data_x, train_data_y)

        predicted_y = decision_forest.predict(test_data_x)


        print(str(est) + "," + str(np.mean(test_data_y == predicted_y)))

#print(np.mean(predicted_y))
#print(np.mean(test_data_y))
#print(np.mean(train_data_y))
