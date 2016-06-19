import cv2
from sklearn.linear_model import SGDClassifier
import numpy as np

feature_det = cv2.FeatureDetector_create("SIFT")
descr_ext = cv2.DescriptorExtractor_create("SIFT")

def preProcessImages(images):
    descriptors= []
    for im in images:
        kpts = feature_det.detect(im)
        kpts, des = descr_ext.compute(im, kpts)
        descriptors.append(des)
    flann_params = dict(algorithm = 1, trees = 5)     
    matcher = cv2.FlannBasedMatcher(flann_params, {})
    bow_extract  =cv2.BOWImgDescriptorExtractor(descr_ext,matcher)
    bow_train = cv2.BOWKMeansTrainer(20)
    for des in descriptors:
        bow_train.add(des)
    voc = bow_train.cluster()
    bow_extract.setVocabulary( voc )
    tdata = []
    for im in images:
        featureset = getImagedata(feature_det,bow_extract,im)
        tdata.extend(featureset)
    return tdata

def getImagedata(feature_det,bow_extract,im):
    featureset = bow_extract.compute(im, feature_det.detect(im))
    return featureset   

def train(traindata,image_classes):  
    clf = SGDClassifier()
    clf.fit(traindata, np.array(image_classes))
    return clf

def test(testimgs, clf):
    answers = []
    testdata = preProcessImages(testimgs)
    for i in testdata:
        answers.append(clf.predict(i.reshape(1,-1)))
    return answers