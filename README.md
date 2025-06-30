# SeismicDataClassification
Supervised learning with Seimsic Data from Montserrat Volcano
Seismographs record tremors on the earth's surface. 
there is an interest in being able to determine the cause of the tremor directly. 
this is how different types of earthquakes can be distinguished. rockfalls are another frequent cause of seismograph deflections, particularly in mountainous and volcanic
regions. 
this is the case on the island of montserrat in the caribbean, for example.

A dataset with already classified vibration events at three different measuring
stations on that island is presented. several hundred of these have been categorized as 
floats over a short discrete period of time. i would like to use this dataset as an example to train
basic machine learning models.

The data is structured as follows:

```
hy_jan/
hy_may/
rf_jan/
rf_may/
```

where each of those four indigrients consist of several hundret subfolder, which in turn consist of .ASC files. In these, the amplitude of the seismograph from three differen researchs station is needed. 


![alt text](https://github.com/OliEhlbeck/SeismicDataClassification/blob/fe8f0ebaea4182114427835e18efdc844a66a19d/Vorticity2D.jpg?raw)

