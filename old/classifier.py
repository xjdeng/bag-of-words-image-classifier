import RootSIFT as r
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import RandomizedPCA
import cv2

def classToNum(mymap,classes):
    output = []
    for i in classes:
        output.append(mymap.index(i) + 0.0)
    return np.array(output)

def numToClass(mymap,nums):
    output = []
    for i in nums:
        output.append(mymap[int(i)])
    return output



    

class classifier(object):
    
    def __init__(self, classmap, clf=LinearSVC(C=100), k=100):
        self.classmap = classmap
        self.clf = clf
        self.k = k
        self.kreg = None
        self.features = None
        self.num_predictions = None
        self.PCA_components = 0
        self.n_features = 0
        self.pca = None
    
    def getFeatures2(self,imgs):
        n = len(imgs)
        feature_det = cv2.FeatureDetector_create("ORB")
        descr_ext = cv2.DescriptorExtractor_create("ORB")
        flann_params = dict(algorithm = 1, trees = 5)     
        matcher = cv2.FlannBasedMatcher(flann_params, {})
        bow_extract  =cv2.BOWImgDescriptorExtractor(descr_ext,matcher)
        descriptors = []
        for im in imgs:
            # (kpts, des) = rs.detectAndCompute(i)
            kpts = feature_det.detect(im)
            kpts, des = descr_ext.compute(im, kpts)
            descriptors.append(des)
        desnum = descriptors[0]
        for i in descriptors:
            desnum = np.vstack((desnum, i))
        #k-means clustering
        # voc, variance = kmeans(desnum, self.k, 1)
        desnum = desnum.astype("float32")
        voc, variance = kmeans(desnum, self.k,2)
        bow_extract.setVocabulary(voc)
        traindata = []
        for im in imgs:
            featureset = bow_extract.compute(im, feature_det.detect(im))
            traindata.extend(featureset)
        #calculate histogram
        myfeatures = StandardScaler().fit_transform(traindata)
        return myfeatures
    
    def getFeatures(self,imgs):
        n = len(imgs)
        rs = r.RootSIFT()
        descriptors = []
        for i in imgs:
            # (kpts, des) = rs.detectAndCompute(i)
            (kpts, des) = rs.orb_run(i)
            descriptors.append(des)
        desnum = descriptors[0]
        for i in descriptors:
            desnum = np.vstack((desnum, i))
        #k-means clustering
        # voc, variance = kmeans(desnum, self.k, 1)
        desnum = desnum.astype("float32")
        if self.kreg is None:
            voc, variance = kmeans(desnum, self.k,2)
            self.kreg = voc
        else:
            voc = self.kreg
        #calculate histogram
        myfeatures = np.zeros((n,self.k),"float32")
        for i in xrange(n):
            words, distance = vq(descriptors[i], voc)
            for w in words:
                myfeatures[i][w] += 1
        #Perform Tf-Idf vectorization
                
        #nbr_occurences = np.sum((myfeatures > 0) * 1, axis = 0)
        #idf = np.array(np.log((1.0*n + 1)) / (1.0*nbr_occurences + 1), 'float32')
        #myfeatures *= idf
        #myfeatures = idf
        
        # Scale the words
        myfeatures = StandardScaler().fit_transform(myfeatures)
        return myfeatures     
    
    def prefit(self, imgs):
        myfeatures = self.getFeatures(imgs)
        self.features = myfeatures
        return myfeatures
    
    def doPCA(self, threshold=0.8, features = None):
        if features is None:
            features = self.features
        (a,b) = features.shape
        pca = RandomizedPCA(n_components=a).fit(features)
        n = 0
        thesum = 0
        while (n < a) & (thesum < threshold):
            thesum += pca.explained_variance_ratio_[n]
            n += 1
        self.PCA_components = n
        self.pca = RandomizedPCA(n_components=n).fit(features)
        self.features = self.pca.transform(features)
        
    
    def mainfit(self, classes, features = None):
        if features is None:
            features = self.features
        self.clf.fit(features, classToNum(self.classmap,classes))
        return self.clf
    
    def predict(self, x_test):
        test_features = self.getFeatures(x_test)
        if self.pca is not None:
            test_features = self.pca.transform(test_features)        
        num_predictions = self.clf.predict(test_features)
        self.num_predictions = num_predictions
    
    def benchmark(self, classes):
        num_classes = classToNum(self.classmap, classes)
        #comp = num_classes == self.num_predictions
        #self.score = 1.0*sum(comp)/len(comp)
        self.score = f1_score(num_classes, self.num_predictions, average="micro")
        return self.score
    
    def keras_benchmark(self, x_test, classes):
        test_features = self.getFeatures(x_test)
        if self.pca is not None:
            test_features = self.pca.transform(test_features)
        self.clf.evaluate(test_features,classToNum(self.classmap,classes))
        
    def keras_fit(self, classes, features = None):
        if features is None:
            features = self.features
        self.clf.fit(features, classToNum(self.classmap,classes), nb_epoch=150, batch_size=10)
    