import os
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.axes_grid1 import ImageGrid
from skimage.feature import local_binary_pattern as lbp
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import KFold as KF
from sklearn.model_selection import cross_val_score

# Here we load the images, and based on the name of the file we identify the
# class, in this case is a number from 1 to 7 that we can find in the seventh
# character of the filename.

data_path = '/home/nico/Workspace/Python/Jaffe/data/'
imgs = []
fs = (12, 12)

filenames = sorted(os.listdir(data_path))
d = [] # vector of classification labels

for img_name in filenames:
    img = plt.imread(data_path + img_name)
    imgs.append(img)
    d.append(int(img_name[6]))

imgs = np.asarray(imgs)
d = np.asarray(d)
indices = np.random.randint(0, len(imgs)-1, 5)
            
# Here are some of the images used in this example:

plt.figure(figsize=fs)

for i,im in enumerate(imgs[indices]):
    plt.subplot(1, 5, i+1)
    plt.imshow(im, cmap='gray')
    plt.xticks(())
    plt.yticks(())
    plt.title('IMG' + str(indices[i]))
plt.show()

# This step involves the process called "feature extraction".
# We will use the well known LBP (Local Binary Patterns).

b = [i for i in range(0,55)]
b.append(255)

lbp_imgs = []
lbp_hists = []

for im in imgs:
    aux = lbp(im, 8, 28, method='default')
    lbp_imgs.append(aux)
    
    aux2, _ = np.histogram(aux, bins=b)
    lbp_hists.append(aux2)

lbp_hists = np.asarray(lbp_hists)

# Here we visulize the respective descriptors of the women shown before.
# (Observation: the last component of each histogram was ommited for
#  visulization purposes, it was too big)

plt.figure(figsize=(12,2))

for i,hist in enumerate(lbp_hists[indices]):
    plt.subplot(1, 5, i+1)
    plt.bar(b[:len(b)-2], hist[:len(hist)-1])
    plt.xticks(())
    plt.yticks(())
    plt.xlim(5)
    plt.title('HIST' + str(indices[i]))
plt.show()

# Now we will try to predict the emotions represented in five random images
# training a simple classifier with the rest. The classifier that we will use
# is the K-Nearest Neighbor (KNN).

# In this step we divide the dataset in training and testing images.
# This can be performed in many ways, the simplest one is K-Fold.
# Here we divide the images in ~30 groups (i.e, 5 images per group),
# this means that we train with ~140 images and test the results with the
# remaining 5 images. This is just an example (very straigforward, but with
# very poor results for the same reason), but in general it is recommended to
# use approximately the 70% of the data for training if you are going to use
# this method to group your data into training/testing sets.

X = np.asarray(lbp_hists)
kf = KF(n_splits=30, shuffle=True).split(X)

train_indices, test_indices =  next(kf)

print('Training images:', train_indices, '\n')
print('Testing images:', test_indices, '\n')

X_train = X[train_indices]
y_train = d[train_indices]

X_test = X[test_indices]
y_test = d[train_indices]

knn = KNN(n_neighbors=1).fit(X_train, y_train)
class_prediction = knn.predict(X_test)

print('Predicted Classes:', class_prediction, '\n')
print('Real Classes:', d[test_indices], '\n')    

# These results are best represented with their respective images and their
# predicted labels:

plt.figure(figsize=fs)

for i,im in enumerate(imgs[test_indices]):
    plt.subplot(1, 5, i+1)
    plt.imshow(im, cmap='gray')
    plt.xticks(())
    plt.yticks(())
    
    emotions = np.array(['Neutral', 'Happy', 'Sad', 'Surprise', 'Disgust',
                         'Anger' , 'Fear'])
    title = str(emotions[class_prediction[i]-1])
    
    if title == str(emotions[d[test_indices][i]-1]):
        plt.title(title, color='green')
    else:
        plt.title(title, color='red')
plt.show()

# Sometimes it is difficult even for us (humans) to identify the emotion that
# is supposedly being expressed. In this images if the real emotion fits the
# predicted one, then the title is green (right prediction), red otherwise
# (wrong prediction).

# Finally we can make some cross validations to get a better feeling of how our
# classification is performing:

knn = KNN(n_neighbors=1)
score_knn = cross_val_score(knn, X, d, cv=10)
print('KNN MEAN PERFORMANCE: ',str(np.mean(score_knn)*100)[:5] + '%')
