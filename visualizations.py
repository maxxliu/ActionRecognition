# visualizations for our results so far
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


# Test 1: Optical flow
'''
Dataset: 1544641757
Frames/Video: 5
Frame Dimensions: 128x128
CNN Layers: 3
'''
trn_acc1 = [0.1096, 0.2986, 0.4960, 0.6477, 0.7479, 0.8056, 0.8445, 0.8703, 0.8906, 0.9055, 0.9153, 0.9235, 0.9319, 0.9368, 0.9386]
val_acc1 = [0.2235, 0.3639, 0.4606, 0.4992, 0.5260, 0.5443, 0.5616, 0.5648, 0.5750, 0.5756, 0.5779, 0.5900, 0.5939, 0.5982, 0.6024]
test1 = 0.6160


# Test 2
'''
Dataset: 1544595196
Frames/Video: 5
Frame Dimensions: 128x128
CNN Layers: 3
'''
trn_acc2 = [0.1052, 0.2829, 0.4282, 0.5187, 0.5766, 0.6281, 0.6604, 0.6910, 0.7135, 0.7353, 0.7579, 0.7767, 0.7858, 0.8012, 0.8095]
val_acc2 = [0.2244, 0.4090, 0.5113, 0.5505, 0.5831, 0.6103, 0.6135, 0.6299, 0.6380, 0.6527, 0.6531, 0.6501, 0.6648, 0.6700, 0.6678]
test2 = 0.6680


# Test 3
'''
Dataset: 1544588764
Frames/Video: 5
Frame Dimensions: 125x125
CNN Layers: 3
'''
trn_acc3 = [0.1394, 0.3729, 0.5190, 0.6180, 0.6881, 0.7372, 0.7789, 0.8079, 0.8306, 0.8524, 0.8633, 0.8769, 0.8875, 0.8956, 0.8979]
val_acc3 = [0.3156, 0.4897, 0.5606, 0.6181, 0.6488, 0.6792, 0.6740, 0.6896, 0.6913, 0.6913, 0.7050, 0.6988, 0.6994, 0.7089, 0.7014]
test3 = 0.6948


# Test 4
'''
Dataset: 1544666342
Frames/Video: 10
Frame Dimensions: 125x125
CNN Layers: 2
'''
trn_acc4 = [0.2249, 0.5006, 0.6184, 0.6887, 0.7372, 0.7722, 0.7981, 0.8151, 0.8318, 0.8462, 0.8570, 0.8655, 0.8713, 0.8803, 0.8883]
val_acc4 = [0.4601, 0.6168, 0.6691, 0.7212, 0.7334, 0.7566, 0.7723, 0.7749, 0.7862, 0.7842, 0.7877, 0.7932, 0.7971, 0.8043, 0.8032]
test4 = 0.8195


# Test 5
'''
Dataset: 1544666342
Frames/Video: 10
Frame Dimensions: 125x125
CNN Layers: 3
'''
trn_acc5 = [0.1771, 0.3952, 0.5058, 0.5744, 0.6247, 0.6613, 0.6910, 0.7136, 0.7324, 0.7476, 0.7623, 0.7712, 0.7789, 0.7949, 0.8014]
val_acc5 = [0.3616, 0.5368, 0.6214, 0.6784, 0.7220, 0.7463, 0.7674, 0.7875, 0.7914, 0.8076, 0.8195, 0.8272, 0.8295, 0.8316, 0.8383]
test5 = 0.8503


# Test 6
'''
Dataset: 1544666342
Frames/Video: 10
Frame Dimensions: 125x125
CNN Layers: 4
'''
trn_acc6 = [0.1286, 0.2960, 0.4107, 0.4848, 0.5343, 0.5730, 0.6022, 0.6259, 0.6445, 0.6620, 0.6791, 0.6908, 0.7063, 0.7167, 0.7266]
val_acc6 = [0.2463, 0.4167, 0.5198, 0.5967, 0.6457, 0.6689, 0.7200, 0.7390, 0.7558, 0.7702, 0.7824, 0.7991, 0.8081, 0.8169, 0.8216]
test6 = 0.8306


# Test 7
'''
Dataset: 1544676882
Frames/Video: 14
Frame Dimensions: 125x125
CNN Layers: 3
'''
trn_acc7 = [0.2419, 0.5084, 0.6139, 0.6751, 0.7116, 0.7416, 0.7644, 0.7785, 0.7955, 0.8058, 0.8182, 0.8262, 0.8341, 0.8417, 0.8472]
val_acc7 = [0.5093, 0.6699, 0.7472, 0.7934, 0.8131, 0.8406, 0.8496, 0.8581, 0.8700, 0.8707, 0.8818, 0.8897, 0.8899, 0.8979, 0.8990]
test7 = 0.8971


###########################
# Visualizing the results #
###########################
plt.figure(); plt.title("Validation accuracy across epochs", fontsize=20)
plt.xlabel("Epoch"); plt.ylabel("Accuracy")
val_results = [val_acc1, val_acc2, val_acc3, val_acc4, val_acc5, val_acc6, val_acc7]
val_results = np.array([np.array(i) for i in val_results])
data = val_results.T
df = pd.DataFrame(data, epochs, ["Test 1", "Test 2", "Test 3", "Test 4", "Test 5", "Test 6", "Test 7"])
ax = sns.lineplot(data=df, dashes=False)
plt.show()

plt.figure(); plt.title("Training accuracy across epochs", fontsize=20)
plt.xlabel("Epoch"); plt.ylabel("Accuracy")
trn_results = [trn_acc1, trn_acc2, trn_acc3, trn_acc4, trn_acc5, trn_acc6, trn_acc7]
trn_results = np.array([np.array(i) for i in trn_results])
data = trn_results.T
df = pd.DataFrame(data, epochs, ["Test 1", "Test 2", "Test 3", "Test 4", "Test 5", "Test 6", "Test 7"])
ax = sns.lineplot(data=df, dashes=False)
plt.show()

plt.figure(); plt.title("Validation outperforms training", fontsize=20)
plt.xlabel("Epoch"); plt.ylabel("Accuracy")
t7_results = [trn_acc7, val_acc7]
t7_results = np.array([np.array(i) for i in t7_results])
data = t7_results.T
df = pd.DataFrame(data, epochs, ["Test 7 Training", "Test 7 Validation"])
ax = sns.lineplot(data=df)
plt.show()
