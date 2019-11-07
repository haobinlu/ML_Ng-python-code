#%%

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
from sklearn import svm


path1 = 'data/ex6data1.mat'
data1 = sio.loadmat(path1)
data = pd.DataFrame(data1['X'], columns=['X1', 'X2'])
data['Y'] = data1['y']


# def plot(data):
negtive = data[data['Y'].isin([0])]
postive = data[data['Y'].isin([1])]
# fig, ax = plt.subplots(figsize= (12,8))
plt.figure(num= 1, figsize=(12, 8))
# ax = fig.add_subplot(111)
# ax.scatter(negtive['X1'], negtive['X2'], marker= 'o', c= 'yellow')
# ax.scatter(postive['X1'], postive['X2'], marker= '+', c= 'black')
plt.scatter(negtive['X1'], negtive['X2'], marker= 'o', c= 'yellow')
plt.scatter(postive['X1'], postive['X2'], marker= '+', c= 'black')
plt.ion()

svc = svm.LinearSVC(C=1, loss='hinge', max_iter=1000)
svc.fit(data[['X1', 'X2']], data['Y'])
svc.score(data[['X1', 'X2']], data['Y'])
coef = svc.coef_
intercept = svc.intercept_
print(coef[0][0])
# plot(data)
plt.figure(num= 1)
# fig1 = plt.figure(num='2', figsize=(12,8),dpi=72)
plot_X = np.linspace(-1, 5, 100)
# ax1 = fig1.add_subplot(111, sharex=ax, sharey= ax)
plt.plot(plot_X, -(plot_X*coef[0][0] + intercept) / coef[0][1], c= 'blue')

plt.pause(15)

#%%



