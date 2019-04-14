# Iceberg
Test task: hockey players classification
# Data cleaning
The first thing that we should to do is just look at the labeled images that we have. It may be noticed that the images numbered 22, 46, 69, 113, 140, 150, 171 contains players from both teams and must be removed from the training set.
# Image resizing
After cleaning we should bring images to uniform size to speed up learning. The large size of all images is 128, so we will resize them to 128x128 size.
# Data augmentation
The size of training set is very small, so we need to do image augmentation to increase it. We will rotate, make horizontal rotate and do some shifts.
# Model
We will use simple convolutional neural network (CNN), because it is the most suitable NN for image classification.
# Model evaluation
This is binary classification task and classes are balanced so we can use metric "accuracy". Of course, F1 metric is more powerful but under such conditions its use is inexpedient.