import pickle
from sklearn.ensemble import GradientBoostingClassifier
from data_loader import load_data

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


X, X_test, Y, Y_test = load_data(use_z_only = True)

# Hyperparameter tuning
print("Hyperparameter tuning ... ")
param_grid = {
    'n_estimators': [60, 70],
    'max_depth': [5,6],
    'learning_rate': [0.01,0.1],
}
print("\t Grid search ... ")
classifier = GradientBoostingClassifier()
grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=3, n_jobs=-1, verbose=0, scoring='accuracy')
gdm_model = grid_search.fit(X, Y)
best_par = grid_search.best_params_

# Train model

#best_par = {'n_estimators': 70,
#    'max_depth': 6,
#    'learning_rate': 0.1
#           }

print("Train model ... ")
gdm_model = GradientBoostingClassifier(n_estimators=best_par['n_estimators'],
                                       max_depth = best_par['max_depth'],
                                       learning_rate =best_par['learning_rate'])
gdm_model.fit(X, Y)

print("Best parameters found: ")
print(best_par)

# Evaluate model
print("Evaluate model ... ")
prediction = gdm_model.predict(X_test)
cm = confusion_matrix(Y_test, prediction)
print(cm)
print("Accuracy:",  accuracy_score(Y_test, prediction))

# Save model to file
target_folder = "Models"
target_file = target_folder + "/gdm_model.pkl"
print(f"Saving dataset to {target_file}...")
with open(target_file, 'wb') as f:
    pickle.dump(gdm_model, f)
