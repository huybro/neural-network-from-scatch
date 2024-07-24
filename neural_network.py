import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer



class NeuralNetwork:
    def __init__(self, layer_sizes, initial_weights =None):
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        if initial_weights:
            self.weights = initial_weights
        else:
            self.weights = [np.random.randn(layer_sizes[l], layer_sizes[l-1]+1) for l in range(1, self.num_layers)]        
        self.neurons = [np.ones(i + 1) for i in layer_sizes]

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward_propagation(self, instance):
        self.neurons[0][1:] = instance  # Set input neurons (excluding bias term)
        print(f"a1: {self.neurons[0]}")
        for l in range(1, self.num_layers):
            # Compute the weighted sum of the previous layer's activations
            z = np.dot(self.weights[l-1], self.neurons[l-1])
            # Apply the activation function to the weighted sum
            self.neurons[l][1:] = self.sigmoid(z)
            print(f"z{l+1}: {z}")
            print(f"a{l+1}: {self.neurons[l]}")
        return self.neurons[-1][1:]  # Return the activations of the output layer

    def cost_function(self, instances, labels, lambda_):
        m = len(instances)
        outputs = []
        for i in instances:
            output = self.forward_propagation(i)
            outputs.append(output)  

        outputs = np.array(outputs)
        labels = np.array(labels)  # Convert labels to numpy array
        J = -np.sum(labels * np.log(outputs) + (1 - labels) * np.log(1 - outputs)) / m
        print(f"Cost, J, associated with this instance: {J:.5f}")
        S = 0
        for w in self.weights:
            S += np.sum(w[:, 1:] ** 2)
        S = (lambda_ / (2 * m)) * S

        return J + S

    def back_propagation(self, instances, labels, lambda_, learning_rate, epochs, tolerance=None):
        m = len(instances)
        total_gradients = [np.zeros(w.shape) for w in self.weights]
        costs = []

        for epoch in range(epochs):
            epoch_gradients = [np.zeros(w.shape) for w in self.weights]

            for instance, label in zip(instances, labels):
                gradients = [np.zeros(w.shape) for w in self.weights]
                self.forward_propagation(instance)
                delta = self.neurons[-1][1:] - np.array(label)
                print(f"delta{self.num_layers}: {delta}")
                gradients[-1] += np.dot(delta.reshape(-1, 1), self.neurons[-2].reshape(1, -1))
                epoch_gradients[-1] += np.dot(delta.reshape(-1, 1), self.neurons[-2].reshape(1, -1))
                print(f"Gradients of Theta{self.num_layers-1} based on training instance \n {gradients[-1]}")

                for k in range(self.num_layers - 2, 0, -1):
                    delta = np.dot(self.weights[k][:, 1:].T, delta) * self.neurons[k][1:] * (1 - self.neurons[k][1:])
                    print(f"delta{k+1}: {delta}")
                    gradients[k-1] += np.dot(delta.reshape(-1, 1), self.neurons[k-1].reshape(1, -1))
                    epoch_gradients[k-1] += np.dot(delta.reshape(-1, 1), self.neurons[k-1].reshape(1, -1))
                    print(f"Gradients of Theta{k} based on training instance \n {gradients[k-1]}\n")

            print("The entire training set has been processed. Computing the average (regularized) gradients:")
            for k in range(self.num_layers - 1):
                regularizer = lambda_ * self.weights[k]
                regularizer[:, 0] = 0
                epoch_gradients[k] = (1 / m) * (epoch_gradients[k] + regularizer)
                print(f"Final regularized gradients of Theta{k+1}: \n {epoch_gradients[k]}")

            for k in range(self.num_layers - 1):
                self.weights[k] -= learning_rate * epoch_gradients[k]

            # Evaluate the performance of the network
            cost = self.cost_function(instances, labels, lambda_)
            costs.append(cost)

            # Check stopping criterion based on tolerance
            if tolerance and epoch > 0:
                cost_change = abs(costs[-1] - costs[-2])
                if cost_change < tolerance:
                    print(f"Cost change {cost_change:.6f} is below the tolerance threshold {tolerance}. Stopping training.")
                    break
        return costs[0]

def normalize_dataset(file_path):
    # Read the CSV file and rearrange the columns
    data = pd.read_csv(file_path, delimiter='\t')
    #data = data.iloc[:, list(range(1, len(data.columns))) + [0]]

    # Convert the DataFrame to a list of lists (dataset)
    dataset = data.values.tolist()
    """for row in dataset:
        row[-1] = int(row[-1]) - 1"""
    # Separate the features (X) and labels (y)
    X = [row[:-1] for row in dataset]
    y = [row[-1] for row in dataset]
    # Normalize the feature matrix X    
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)


    # Combine the normalized features and labels
    dataset_normalized = [list(X_normalized[i]) + [y[i]] for i in range(len(X))]

    return dataset_normalized

def stratified_cross_validation(dataset, layer_sizes, lambda_, learning_rate, num_epochs,tolerance ,k_folds=10):
    # Shuffle the dataset
    random.shuffle(dataset)

    # Stratify the dataset based on class labels
    stratified_dataset = {}
    for instance in dataset:
        label = instance[-1]
        if label not in stratified_dataset:
            stratified_dataset[label] = []
        stratified_dataset[label].append(instance)

    # Split the stratified dataset into k folds
    folds = [[] for _ in range(10)]
    for label, instances in stratified_dataset.items():
        fold_size = len(instances) // k_folds
        for i in range(k_folds):
            fold = instances[i * fold_size:(i + 1) * fold_size]
            if i < len(instances) % k_folds:
                fold.append(instances[k_folds * fold_size + i])
            folds[i].extend(fold)

    fold_metrics = []
    for i in range(k_folds):
        train_dataset = []
        test_dataset = folds[i]
        for j in range(k_folds):
            if j != i:
                train_dataset.extend(folds[j])

        # Train the neural network
        nn = NeuralNetwork(layer_sizes)
        train_instances = [instance[:-1] for instance in train_dataset]
        train_labels = [[1 if instance[-1] == j else 0 for j in range(layer_sizes[-1])] for instance in train_dataset]

        # Train the neural network on the entire training set
        nn.back_propagation(train_instances, train_labels, lambda_, learning_rate, num_epochs, tolerance)

        # Evaluate the neural network on the test set
        test_predictions = []
        for instance in test_dataset:
            input_data = instance[:-1]
            output = nn.forward_propagation(input_data).tolist()
            predicted_label = output.index(max(output))
            test_predictions.append(predicted_label)

        test_labels = [instance[-1] for instance in test_dataset]

        accuracy, precision, recall, f1 = compute_metrics(test_labels, test_predictions)
        fold_metrics.append((accuracy, precision, recall, f1))

    avg_accuracy = sum(acc for acc, _, _, _ in fold_metrics) / len(fold_metrics)
    avg_precision = sum(prec for _, prec, _, _ in fold_metrics) / len(fold_metrics)
    avg_recall = sum(rec for _, _, rec, _ in fold_metrics) / len(fold_metrics)
    avg_f1 = sum(f1 for _, _, _, f1 in fold_metrics) / len(fold_metrics)

    return avg_accuracy, avg_f1

def compute_metrics(true_labels, predicted_labels):
    accuracy = sum(1 for pred, label in zip(predicted_labels, true_labels) if pred == label) / len(true_labels)

    class_metrics = {}
    for class_label in set(true_labels):
        true_positive = sum(1 for pred, label in zip(predicted_labels, true_labels) if pred == label == class_label)
        predicted_positive = sum(1 for pred in predicted_labels if pred == class_label)
        precision = true_positive / predicted_positive if predicted_positive != 0 else 0
        recall = true_positive / sum(1 for label in true_labels if label == class_label)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        class_metrics[class_label] = (precision, recall, f1)

    overall_precision = sum(precision for precision, _, _ in class_metrics.values()) / len(class_metrics)
    overall_recall = sum(recall for _, recall, _ in class_metrics.values()) / len(class_metrics)
    overall_f1 = sum(f1 for _, _, f1 in class_metrics.values()) / len(class_metrics)

    return accuracy, overall_precision, overall_recall, overall_f1


def plot(dataset,layer_sizes,lambda_reg,name):
    costs = []
    nn = NeuralNetwork(layer_sizes,None)

    X = [instance[:-1] for instance in dataset]  # Extract the feature columns
    y = [instance[-1] for instance in dataset]  # Extract the target column
    # Split the dataset into training and testing sets
    training_set, testing_set = train_test_split(dataset, test_size=0.3, stratify=y)
    train_instances = [instance[:-1] for instance in training_set]
    train_labels = [[1 if instance[-1] == j else 0 for j in range(layer_sizes[-1])] for instance in training_set]
    testing_instances = [instance[:-1] for instance in testing_set]
    testing_labels = [[1 if instance[-1] == j else 0 for j in range(layer_sizes[-1])] for instance in testing_set]
    for i in range(len(train_instances)):
        nn.back_propagation(train_instances[i],train_labels[i],lambda_reg,1,1,1e-7)
        costs.append(nn.cost_function(testing_instances,testing_labels,0))
    
    plt.plot(range(1, len(costs) + 1), costs)
    plt.xlabel('Number of Training Instances')
    plt.ylabel('Cost (J) on Test Set')
    plt.title(f'Learning Curve for Neural Network for {name} Dataset')
    plt.show()
# Example usage
if __name__ == "__main__":

    def example_testing(): 
        # Read the network architecture and initial weights
        #backprop_example1
        """lambda_reg = 0
        layer_sizes = [1,2,1]
        initial_weights = [ np.array([[0.4, 0.1], [0.3, 0.2]]), np.array([[0.7, 0.5, 0.6]]) ] 
        # Training data
        instances = [[0.13],[0.42]]
        labels = [[0.9],[0.23]]"""


        #backprop_example2
        lambda_reg = 0.250
        layer_sizes = [2,4,3,2]
        initial_weights = [
            np.array([[0.42000, 0.15000, 0.40000], [0.72000,  0.10000,  0.54000],[0.01000 , 0.19000 , 0.4200],[0.30000 , 0.35000 , 0.68000]]),
            np.array([[	0.21000 , 0.67000 , 0.14000 , 0.96000 ,0.87000],[0.87000 , 0.42000, 0.20000 , 0.32000 , 0.89000],[0.03000 , 0.56000  ,0.80000  ,0.69000 , 0.09000]]),
            np.array([[	0.04000,  0.87000  ,0.42000,  0.53000 ],[0.17000 , 0.10000 , 0.95000 , 0.69000]])
        ]
        # Training data
        instances = [[0.32, 0.68],[0.83, 0.02]]
        labels = [[0.75, 0.98], [0.75, 0.28]]
        

        # Create the NeuralNetwork instancec
        nn = NeuralNetwork(layer_sizes, initial_weights)



        for instance, label in zip(instances, labels):
                print(f"Forward propagating the input {instance}")
                output = nn.forward_propagation(instance)
                print(f"Predicted output for this instance: {output}")
                print(f"Expected output for this instance: {label}")
                
                cost = nn.cost_function([instance], [label], lambda_reg)

        # Print final cost
        cost = nn.cost_function(instances, [label for label in labels], lambda_reg)
        print(f"\nFinal (regularized) cost, J, based on the complete training set: {cost:.5f}")

        # Backpropagation and print outputs
        print("\n--------------------------------------------")
        print("\nRunning backpropagation")
        nn.back_propagation(instances, labels, lambda_reg, learning_rate=1, epochs=1, tolerance=1e-5)

    #Uncomment this function to verify the correctness
    example_testing()

    
    
    """data = pd.read_csv('datasets/hw3_house_votes_84.csv')
    X = data.drop(columns=['class']).values  # Features
    y = data['class'].values  # Labels
    encoder = OneHotEncoder()
    X_encoded = encoder.fit_transform(X)
    y = y.reshape(-1, 1)
    dataset = np.concatenate((X_encoded.toarray(), y), axis=1)"""



   # Read the dataset
    """data = pd.read_csv('datasets/contraceptive+method+choice (1)/cmc.csv')
    categorical_cols = [1, 2, 6, 7, 4, 5, 8]
    numerical_cols = [0, 3, 9]
    # Specify categorical and numerical columns
    encoder = OneHotEncoder()
    X_categorical = data.iloc[:, categorical_cols]
    X_categorical_encoded = encoder.fit_transform(X_categorical)

    X_numerical = data.iloc[0:, numerical_cols].values
    X = np.concatenate((X_categorical_encoded.toarray(), X_numerical), axis=1)
    for row in X:
        row[-1] = int(row[-1]) - 1"""