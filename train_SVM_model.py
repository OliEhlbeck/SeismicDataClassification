import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import svm
from data_loader import load_data
from plot_cm import plot_confusion_matrix

X, X_test, Y, Y_test = load_data(use_z_only = True)

parameters = {
    'kernel':('linear', 'rbf'),
    'C':[1, 10]
}

svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(X, Y)
best_parameter = clf.best_params_
sorted(clf.cv_results_.keys())

# Train model
print("Train model ... ")
best_parameter= {
    'kernel': 'linear',
    'C':10
}
svc_model = svm.SVC(kernel = 'linear', C = 10)
svc_model.fit(X, Y)

# Evaluate model svm
print("Evaluate model ... ")
prediction = svc_model.predict(X_test)
plot_confusion_matrix(Y_test, prediction)

# Accuracy
print(f"Accuracy: {accuracy_score(Y_test, prediction)}")

# Save model to file
print("Save model to file ... ")
with open('gdm_model.pkl', 'wb') as f:
    pickle.dump(gdm_model, f)

