#!/usr/bin/python
from __future__ import print_function
import sys
import numpy as np
import theano
import theano.tensor as T
from theano import config, shared
import pickle

#config.optimizer = 'fast_compile'
nDims = 7
nPredictors = 3
nHidden = 100
rate = .01
nEpochs = 200
wt_decay = .0
area_decay = .0
var = nDims

rootndims = float(nDims) ** .5
sqrndims = nDims ** 2

with open('blah.pkl', 'rb') as f:
    corpus = pickle.load(f)
corpus = np.array(corpus, dtype='int32')
nCorpus = len(corpus)
nWords = corpus.max()+1
print(corpus.min(), nWords, nCorpus)

lexicon = np.random.normal(size=(nWords, nDims))
for word in lexicon:
    word /= np.linalg.norm(word) / rootndims


def share(data, dtype=config.floatX):
    return shared(np.asarray(data, dtype), borrow=True)


lexicon = share(lexicon)


def print_lex():
    for word in lexicon.get_value():
        neighbour = np.mean([np.linalg.norm(word - word2) for word2 in lexicon.get_value()])
        print('({:7.4} {:7.4} {:7.4} {:7.4}), '.format(
            word.max(), word.min(), np.linalg.norm(word), neighbour), end='\n')

#print_lex()

inpt = T.ivector()
output = T.iscalar()
predictors = lexicon[inpt].flatten()


def get_wts(n_in, n_out, wname='W', bname='b'):
    w_values = np.asarray(
        np.random.uniform(low=-1, high=1, size=(n_in, n_out)), dtype=config.floatX)
    w = shared(w_values, name=wname, borrow=True)
    b_values = np.asarray(
        np.random.uniform(low=-1, high=1, size=n_out), dtype=config.floatX)
    b = shared(b_values, name=bname, borrow=True)
    return w, b


W1, b1 = get_wts(nPredictors * nDims, nHidden, 'W1', 'b1')
hidden = T.tanh(T.dot(predictors, W1) + b1)

W2, b2 = get_wts(nHidden, nDims, 'W2', 'b2')
output_vec = (T.dot(hidden, W2) + b2)

mse = T.sum((output_vec - lexicon[output]) ** 2)/var
others = T.sum((lexicon-output_vec)**2, axis=1)/var
prob = T.exp(-mse)/T.sum(T.exp(-others))

lex1 = lexicon.dimshuffle('x', 0, 1)
lex2 = lexicon.dimshuffle(0, 'x', 1)
area = T.mean((lex1 - lex2) ** 2)

wt_cost = 0
for param in (W1, b1, W2, b2):
    wt_cost += (param ** 2).sum()

cost = -T.log(prob) #+ wt_decay * wt_cost #- area_decay * area

updates = []
for param in (W1, b1, W2, b2):
    update = param - rate * T.grad(cost, param)
    updates.append((param, update))

lex_update = lexicon - rate * T.grad(cost, lexicon)
lex_norms = T.sqrt(T.sum(T.sqr(lexicon), axis=1, keepdims=True))
lex_update = rootndims * lex_update / lex_norms
updates.append((lexicon, lex_update))

print('Compiling them eggs...')
trainer = theano.function([inpt, output], cost, updates=updates)
tester = theano.function([inpt, output], prob)
areaer = theano.function([], [area, wt_cost])

print('\n\nepoch, cost , prob , area , wt_cost ')

for epoch in range(nEpochs):
    prob, cost = 0.0, 0.0
    # Use 0.75 for training and 0.25 for test
    percent = 0.1;
    total_num = int(percent*(nCorpus - nPredictors));
    split_point = int(0.75*total_num);
    for i in range(0,split_point):
        print('TR{:6d}'.format(i), end='')
        cost += trainer(corpus[i:i + nPredictors], corpus[i + nPredictors])
        print('\b\b\b\b\b\b\b\b', end='')

    for i in range(split_point,total_num):
        print('TS{:6d}'.format(i), end='')
        prob += tester(corpus[i:i + nPredictors], corpus[i + nPredictors])
        print('\b\b\b\b\b\b\b\b', end='')

    prob /= nCorpus - nPredictors
    cost /= nCorpus - nPredictors
    ar, wt_c = areaer()
    print('{:5}, {:6.4f}, {:6.4f}, {:6.4f}, {:6.4f}'.format(epoch, cost, prob, float(ar), float(wt_c)))

print_lex()

########################### Do Some PCA + K means ##########################

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import datasets

pca = PCA(n_components=3)
pcs = pca.fit_transform(lexicon.get_value())
print('PC shape:', pcs.shape)

np.random.seed(5)

estimator = KMeans(n_clusters=5)
fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
plt.cla()
estimator.fit(pcs)
labels = estimator.labels_

ax.scatter(pcs[:,0], pcs[:,1], pcs[:,2], c=labels.astype(np.float), s=100)

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('1st PC')
ax.set_ylabel('2nd PC')
ax.set_zlabel('3rd PC')

# for label, x, y, z in zip(range(nWords), pcs[:, 0], pcs[:, 1],pcs[:,2]):
#     plt.annotate(
#         str(label),
#         xy = (x, y, z),
#         #xytext = (-20, 20),
#         #textcoords = 'offset points', ha = 'right', va = 'bottom',
#         #bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
#         #arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0')
#     )

plt.show()