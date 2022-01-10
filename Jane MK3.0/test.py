import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

best_p = np.load("data/jan_09_2022/best_p.npy")
best_s = np.load("data/jan_09_2022/best_s.npy")
training_rois = np.load("data/jan_09_2022/training_rois.npy")
validation_rois = np.load("data/jan_09_2022/validation_rois.npy")
#plt.matshow(data)
#plt.show()
#plt.show(training_rois[:100])
plt.plot(training_rois[:100])
plt.plot(validation_rois[:100])

plt.show()

