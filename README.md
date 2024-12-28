# Activity Detection using CNNs

**Abstract**
This project presents the development of an innovative Android-based application
interfaced with the RESpeck sensor, harnessing machine learning to monitor and
classify physical activities and respiratory patterns in real-time. The objective was
to develop a prototype to demonstrate the feasibility of using machine learning
models, particularly TensorFlow Lite, for immediate activity recognition within a
mobile application framework
The methodology involved the integration of the RESpeck sensor with an Android
application, employing Bluetooth Low Energy for data transmission. The application
was constructed in Android Studio, using Kotlin and Java, and was structured to preprocess sensor data before being classified by the TensorFlow Lite model. Testing was
conducted across multiple Android devices to ensure functionality and performance.
Results from the prototype testing indicated successful data communication and classification of activities and critical analysis of the model revealed high testing scores and
in cases of discrepancy, provided explanations on any misclassification that happened
during activity recognition.
In conclusion this prototype represents good progress in mobile application activity
recognition which could be applied to several different systems such as in the fitness
and the healthcare industries, however the latter may have a regulatory barrier to entry
for medical device approval.
Looking forward, with access to higher performance computing or simply more time,
we would be able to increase the rigour of the testing process to improve the model
beyond it’s current capabilities and investigate new heuristics or model changes and
their effects.


**The Model**

The model used a 3D sliding window throughout time across each of the three main input channels.
The RESpeck device has two sensors (gyroscope and accelerometer) each giving readings for 3 axes.
The three input streams each consisted of a (75,6) tensor representing 75 readings across each sensor-axis combination from the raw data, fourier amplitudes, and smoothed differentials.
These channels were each standardised, pushed through a couple of convolutional and pooling layers, then concatenated and sent through some affine layers, before separating out into to output classifciations.
These represented the respiratory acitvity such as hyperventilating/normal breathing, and the movement such as walking/ascending stairs.

More details on the nature of the classification task and the models performance and architectural decisions can be found in the paper in the repository.
