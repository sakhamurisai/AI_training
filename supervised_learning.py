
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import fetch_california_housing


#supervised learning
# linear regression
def practise_models_with_odd_or_even(train_number, test_numbers,model_input):
    print(f"training the model with {train_number} numbers using the algorithm {model_input}")
    x_train_set = np.array([[n ]for n in range(-20,train_number)])
    y_train_set = np.array([n%2 for n in range(-20,train_number)])

    model = model_input
    model.fit(x_train_set, y_train_set)

#sci-kit expects the input to be 2D array
    new_test_set = np.array([[n] for n in range(-10,test_numbers)])
    prediction = model.predict(new_test_set)

    for number, prediction in zip(new_test_set.flatten(), prediction):
        if prediction == 0:
            print(f" test number {number}, then even")
        else:
            print(f"test number {number}, then odd")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    algorithm = [LinearRegression, LogisticRegression, DecisionTreeClassifier]
    for model in algorithm:
        practise_models_with_odd_or_even(10000, 10, model())