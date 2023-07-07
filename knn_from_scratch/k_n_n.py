import numpy as np

class KNN():
    def __init__(self,k):
        self.k = k
        
    def train(self,x,y):
        self.x_train = x
        self.y_train = y
        pass

    def predict(self,x_test,num_loops=2):
        if num_loops == 0:
            distances = self.compute_distance_vecorized(x_test)
        elif num_loops == 1:
            distances = self.compute_distance_one_loop(x_test)
        elif num_loops == 2:
            distances = self.compute_distance_two_loop(x_test)
        return self.predict_labels(distances)

    def compute_distance_vecotrized(self,x_test):
        num_test = x_test[0]
        num_train = self.x_train[0]
        distances = np.zeros((num_test,num_train))
        

        
        pass

    def compute_distance_one_loop(self,x_test):
        num_test = x_test[0]
        num_train = self.x_train[0]
        distances = np.zeros((num_test,num_train))

        for i in range(num_test):
            distances[i,:] = np.sqrt(np.sum((self.x_train-x_test[i,:])**2),axis=1)


    def compute_distance_two_loop(self,x_test):
        num_test = x_test[0]
        num_train = self.x_train[0]
        distances = np.zeros((num_test,num_train))

        for i in range(num_test):
            for j in range(num_train):
                # Computing Euclidean distance
                distances[i,j] = np.sqrt(np.sum((x_test[i,:]-self.x_train[j,:]))**2)
        return distances
        
    def predict_labels(self,distances):
        num_test = distances.shape[0]
        y_pred = np.zeros(num_test)
        # Getting most frequent class
        for i in range(num_test):
            y_indices = np.argsort(distances[i,:])
            k_closest_classes = self.y_train[y_indices[:self.k]].astype(int)
            y_pred[i] = np.argmax(np.bincount(k_closest_classes))

        return y_pred

if __name__ == "__main__":
    x = np.loadtxt("data/data.txt",delimiter=", ")
    y = np.loadtxt("data/targets.txt")

    kn_1 = KNN(k=2)

    kn_1.train()
    y_pred = kn_1.predict(x)

    accuracy = sum(y_pred==y)/y.shape[0]

    print(f'Accuracy:{accuracy}')