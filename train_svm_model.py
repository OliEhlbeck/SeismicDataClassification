import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from data_loader_old import load_data


X, X_test, Y, Y_test = load_data(use_z_only = True)

parameters = {
    'kernel':('linear', 'rbf'),
    'C':[1, 10]
}
svc = svm.SVC()

#Tuning
clf = GridSearchCV(svc, parameters)
clf.fit(X, Y)
best_parameter = clf.best_params_
sorted(clf.cv_results_.keys())

# Train model
print("Train model ... ")
#best_parameter= {
#    'kernel': 'linear',
#    'C':10
#}
svc_model = svm.SVC(kernel = best_parameter['kernel'], C = best_parameter['C'])
svc_model.fit(X, Y)

# Evaluate model svm
print("Evaluate model ... ")
prediction = svc_model.predict(X_test)
print(confusion_matrix(Y_test, prediction))
print(f"Accuracy: {accuracy_score(Y_test, prediction)}")

#Detect most important Features
features_names=[]
#features_names = ["E", "kurt", "skewness", "mean", "median", "std", "min_val", "max_val", "rms", "variance", "range_val"]
#features_names.extend(["N ", "kurt2", "skewness2", "mean2", "median2", "std2", "min_val2", "max_val2", "rms2", "variance2", "range_val2"])
features_names.extend(["Z", "kurt3", "skewness3", "mean3", "median3", "std3", "min_val3", "max_val3", "rms3", "variance3", "range_val3"])

flat_list = [item for sublist in svc_model.coef_ for item in sublist]

plt.barh(features_names, flat_list)
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Bar Plot of Array')
plt.show()

# Save model to file
target_folder = "C:/Users/Oli/PycharmProjects/GeoPhyMl/Models"
target_file = target_folder + "/svm_model.pkl"
print(f"Saving dataset to {target_file}...")
with open(target_file, 'wb') as f:
    pickle.dump(svc_model, f)

