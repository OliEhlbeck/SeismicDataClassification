#from sklearn.model_selection import RandomizedSearchCV
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense

from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import RandomizedSearchCV

#from scikeras.wrappers import KerasClassifier
from data_loader import load_data
from plot_cm import plot_confusion_matrix

# Function to create a simple feedforward neural network model
def create_model(layers=[32, 32], activation='relu', optimizer='adam'):
    model = Sequential()
    for units in layers:
        model.add(Dense(units=units, activation=activation))
    model.add(Dense(units=1, activation='sigmoid'))  # Example for binary classification
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Wrap Keras model so it can be used by scikit-learn
model = KerasClassifier(build_fn=create_model, verbose=0)

# Define hyperparameters to search
param_dist = {
    'layers': [(16,), (32,), (64,), (16, 16), (32, 32), (64, 64)],
    'activation': ['relu', 'sigmoid'],
    'optimizer': ['adam', 'rmsprop']
}

# Perform RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=10,  # Number of parameter settings that are sampled
    cv=3,       # Number of cross-validation splits
    verbose=2,
    random_state=42,
    n_jobs=-1    # Use all available cores
)

# Run the random search
X, X_test, Y, Y_test = load_data(use_z_only = True)
X_train = X
Y_train = Y

random_search.fit(X_train, Y_train)  # X_train and y_train are your training data

# Get the best parameters and their respective scores
print("Best parameters found: ", random_search.best_params_)
print("Best score: ", random_search.best_score_)


