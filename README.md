# SeismicDataClassification

## Background

![alt text](https://github.com/OliEhlbeck/SeismicDataClassification/blob/cec6bd4dd65448b021360727cbfadaa51939d16a/images/Soufriere_Hills.jpg?raw)

Seismographs record ground vibrations caused by various geological events. One key goal in seismology is to automatically determine the cause of these tremors. This enables the distinction between different types of earthquakes and other sources of ground motion. Rockfalls, for example, are a frequent cause of seismograph deflections, especially in mountainous and volcanic regions.

A typical case is the island of Montserrat in the Caribbean, where three seismic monitoring stations continuously record ground motion. The island is home to the Soufrière Hills volcano, whose eruptions in the 1990s forced the relocation of much of the island’s population due to the danger posed by volcanic activity. A dataset containing pre-classified vibration events from these three stations is available. Several hundred events, captured over a short and discrete period of time, have been labeled according to their cause.

In geoscience, data availability can be limited, which poses challenges for building and validating models. This dataset therefore offers a valuable opportunity to set up and training machine learning models for earthquake prognosis.

## Dataset

The data is structured as follows:

```
hy_jan/
hy_may/
rf_jan/
rf_may/
```

where each of those four indigrients consist of several hundret subfolder, which in turn consist of .ASC files. In these, the amplitude of the seismograph from three differen researchs station is needed. 

support vector machine's using only z components of reduced size dataset.

```
Confusion Matrix: 
[[3208  1193]
 [   384 5215]]
Accuracy: 0.8423
```

![alt text](https://github.com/OliEhlbeck/SeismicDataClassification/blob/833f305c697bf8c5c83c30609eb41ae75916a8f9/images/FeaturesSVM.jpg?raw)

With 50.000 files, the accuracy increases significantly; which especially holds for the gradient descent method - just as expected.

![alt text](https://github.com/OliEhlbeck/SeismicDataClassification/blob/3389fc589a78dfd08f12f134aae86efff304ad48/images/ConfusionMatrixGradienDescent.jpg?raw)

gfhj
