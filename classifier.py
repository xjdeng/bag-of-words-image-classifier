import numpy as np
import cv2, random
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import time, image, warnings

class classifier(object):
    #Initializes an image classifier object.  All parameters except clf have their default values set to the optimal values discovered through testing.
    
    # clf: sets the actual classifier that will be used for identifying images. This classifier must implement the partial_fit() method. Legal values:
    #   sklearn.naive_bayes.BernoulliNB
    #   sklearn.linear_model.Perceptron
    #   sklearn.linear_model.SGDClassifier (RECOMMENDED)
    #   sklearn.linear_model.PassiveAggressiveClassifier

    # feature_finder: A string which specifies the algorithm used to get features from an image. Legal values:
    #   "SIFT": Slowest but most accurate. Creates 128 dimensional feature vectors.
    #   "SURF": Slightly faster and a little less accurate than SIFT. Creates 64 dimensional feature vectors.
    #   "ORB" : VERY fast but not as accurate as the other two. Creates 32 dimensional feature vectors.

    # convert_grey: Should the image be converted to greyscale before extracting feature vectors?

    # rootsift: Should we apply the RootSIFT adjustment after extracting features?  
    #   See: http://www.pyimagesearch.com/2015/04/13/implementing-rootsift-in-python-and-opencv/

    # pca_before_kmeans: Whether to run PCA before KMeans or after. Must be set to True.

    # kclusters: # of K-means clusters to use when creating the histogram of features.

    # pca_ratio: We'll be using PCA on the feature vectors but since we don't know the # of dimensions beforehand (depends on the feature_finder variable), we'll
    # specify the fraction of dimensions we'd like to have in the transformed space vs the original space.  For example, if you'd like to have only 1/4 as many 
    # dimensions after PCA as before, set pca_ratio to 0.25 .

    # tfidf: Whether or not to use TF-IDF vectorization before training the image classifier.

    # incremental_threshold: # of new images to queue up before retraining the classifier. To better understand this, imagine this algorithm being implemented in a 
    # mobile app. The user of the app takes a picture, sets its category, and gives it to the app but the app won't immediately train on that image right away since 
    # partial_fit() doesn't work too well incrementally training 1 or 2 images at a time. It waits until it reaches a threshold and has images of at least 2 different 
    # categories before incrementally training on all of them at once. This variable specifies that threshold.
    
    
    def __init__(self, clf, feature_finder="SIFT", convert_grey = True, rootsift=True, pca_before_kmeans = True, kclusters=200, pca_ratio = 0.5, tfidf = False, incremental_threshold = 25):
        self.feature_finder = feature_finder
        self.rootsift = rootsift
        self.kclusters = kclusters
        self.kmeans = None
        self.pca = None
        self.pca_before_kmeans = pca_before_kmeans
        self.pca_ratio= pca_ratio
        self.tfidf = tfidf
        self.clf = clf
        self.clf_used = False
        self.detector = cv2.FeatureDetector_create(feature_finder)
        self.extractor = cv2.DescriptorExtractor_create(feature_finder)
        self.categories = None
        self.descriptors = None
        self.des_list = None
        self.image_clases = None
        self.train_time = 0
        self.test_time = 0
        self.convert_grey = convert_grey
        self.train_queue = []
        self.class_queue = []
        self.incremental_threshold = incremental_threshold
        self.fclf = False
    
    # Performs the RootSIFT adjustment on an image when given its keypoints (kps)    
    # Note: Root SIFT is adapted from http://www.pyimagesearch.com/2015/04/13/implementing-rootsift-in-python-and-opencv/
    def RootSIFT(self, img, kps):
        eps=1e-7
        (kps, descs) = self.extractor.compute(img, kps)
        descs = descs.astype(dtype="float32")
        if len(kps) == 0:
            return ([], None)
        descs /= (descs.sum(axis=1, keepdims=True) + eps)
        descs = np.sqrt(descs)
        return (kps, descs)
    
    # Converts an image to grey    
        
    def convertGrey(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   
        
    # Performs TF-IDF vectorization
    
    def tf_idf(self, im_features):
        nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
        idf = np.array(np.log((1.0*len(im_features)+1) / (1.0*nbr_occurences + 1)), 'float32')
        return im_features*idf     
        
        
    # Performs Incremental PCA when training images. Sets up the Incremental PCA environment if it hasn't already been set up.

    def PCA_train(self):
        pcafun = None
        if self.pca == None:            
            (a,b) = self.descriptors.shape
            self.pca = IncrementalPCA(n_components = int(b*self.pca_ratio))
            pcafun = self.pca.fit
        else:
            pcafun = self.pca.partial_fit
        pcafun(self.descriptors)
        self.PCA_common()
    
    # Performs Incremental PCA when testing images.
    
    def PCA_test(self):
        self.PCA_common()
        
    # Helper function for Incremental PCA calling the procedures common between both training and testing.
            
    def PCA_common(self):
        self.descriptors = self.pca.transform(self.descriptors)
        if self.pca_before_kmeans:
            tmp = []
            for i in self.des_list:
                tmp.append(self.pca.transform(i))
            self.des_list = tmp
    
    # Performs Mini Batch K-Means when training images. Sets up the K-Means environment if it hasn't already been set up.
    
    def KMeans_train(self):
        clusterfun = None
        special = False
        newclusters = self.kclusters
        if self.kmeans is None:
            if self.descriptors.shape[0] < self.kclusters:
                special = True
                newclusters = self.descriptors.shape[0]
            self.kmeans = MiniBatchKMeans(n_clusters = newclusters)
            clusterfun = self.kmeans.fit
        else:
            clusterfun = self.kmeans.partial_fit
        clusterfun(self.descriptors)
        if special:
            self.kmeans.set_params(n_clusters = self.kclusters)
        self.KMeans_common()
    
    # Performs Mini Batch K-Means when testing images. 
    
    def KMeans_test(self):
        self.KMeans_common()
    
    # Helper function for Mini Batch K-Means calling the procedures common between both training and testing. Also generates the histogram of image features.
    
    def KMeans_common(self):
        n = len(self.des_list)
        im_features = np.zeros((n, self.kclusters), "float32")
        for i in xrange(n):
            words = self.kmeans.predict(self.des_list[i])
            for w in words:
                im_features[i][w] += 1
        if self.tfidf is True:
            im_features = self.tf_idf(im_features)
        stdSlr = StandardScaler().fit(im_features)
        im_features = stdSlr.transform(im_features)
        self.descriptors = im_features        

    # Helper function that extracts the descriptors and correct image classes from a list of images and stores them in temporary variables in the clasifier
    # for later processing.                          

    def getDescriptors(self, imglist):
        image_classes = []
        des_list = []
        for i in imglist:
            img = i.data
            if self.convert_grey:
                img = self.convertGrey(img)
            kps = self.detector.detect(img)
            (kps, descs) = self.extractor.compute(img, kps)
            descs = descs.astype(dtype="float32")
            if self.rootsift:
                (kps, descs) = self.RootSIFT(img, kps)
            des_list.append(descs)
            image_classes.append(i.category)
        descriptors = des_list[0]
        for descriptor in des_list[1:]:
            descriptors = np.vstack((descriptors, descriptor))
        (self.descriptors, self.des_list, self.image_classes) = (descriptors, des_list, np.array(image_classes))
        
        
    # Adds a list of images to the queue to be trained.  If the queue is over the threshold, then training will begin.

    
    def incremental_train(self, imglist, threshold = None):
        if threshold is None:
            threshold = self.incremental_threshold
        random.shuffle(imglist)
        tmptime = time.time()
        for i in imglist:
            self.partial_train(i, threshold)
        self.train_time = time.time() - tmptime
        
    # Helper function that adds a single image to the queue and will call train() if the threshold is reached.
    
    
    def partial_train(self, img, threshold):
        self.train_queue.append(img)
        self.class_queue.append(img.category)
        self.class_queue = list(set(self.class_queue))
        if (len(self.class_queue) > 1) & (len(self.train_queue) >= threshold):
            self.train(self.train_queue)
            self.train_queue = []
            self.class_queue = []
            
    # Helper function for trainling a list of images.  Guides through feature extraction, PCA, K-Means clustering, hisogram generation, and fitting the histogram.
    # Please don't call this function directly; use incremental_train() whenever adding new training images.

    
    def train(self, imglist):
        self.train_time = time.time()
        self.getDescriptors(imglist)
        if self.pca_before_kmeans is True:
            self.PCA_train()
            self.KMeans_train()
        else:
            self.KMeans_train()
            self.PCA_train()
        if self.clf_used is True:
            fitfun = self.clf.partial_fit
        else:
            fitfun = self.clf.fit
        fitfun(self.descriptors, self.image_classes)
        self.train_time = time.time() - self.train_time
    
    # Takes a list of images and predicts the classes they belong to and compares those predicted classes with their actual classes which, in turn, is benchmarked 
    # in the form of an F1 score.
    
    def test(self, test_cases):
        self.test_time = time.time()
        answer_list = []
        for i in test_cases:
            answer_list.append(i.category)
        answers = np.array(answer_list)
        self.getDescriptors(test_cases)
        if self.pca_before_kmeans is True:
            self.PCA_test()
            self.KMeans_test()
        else:
            self.KMeans_test()
            self.PCA_test()
        predictions = self.predict()
        self.test_time = time.time() - self.test_time
        return self.benchmark(answers=answers, predictions=predictions)
    
    
    # Helper function which takes the descriptors (expressed in histogram form) stored in the classifier and predicts the image classes from them.    
    
    def predict(self):
        return self.clf.predict(self.descriptors)
    
    # Takes the path to an image, predits its class, and "writes" the class onto the image and outputs the result as a new image at the path specified in newimg.
    
    def predict_and_visualize(self,imgpath,newimg):
        img = image.image(imgpath,"None")
        self.predict_and_visualize2(img,newimg)
    
    # Helper function which works like predict_and_visualize() except it takes an image object instead of the image itself as an input.
    
    def predict_and_visualize2(self,img,newimg):
        dummyimg = image.image("dummyimage.jpg","None")
        dummyimg2 = image.image("dummyimage2.jpg","None")
        dummyimg3 = image.image("dummyimage3.jpg","None")
        dummyimg4 = image.image("dummyimage4.jpg","None")
        if img.data.shape[0] > 600:
            b = img.data.shape[1]*600/img.data.shape[0]
            img.data = cv2.resize(img.data,(b,600))
        self.getDescriptors([dummyimg, dummyimg2, img, dummyimg3, dummyimg4])
        if self.pca_before_kmeans is True:
            self.PCA_test()
            self.KMeans_test()
        else:
            self.KMeans_test()
            self.PCA_test()
        prediction = self.predict()
        ptext = prediction[2]
        cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
        pt = (0,3*img.data.shape[0] // 4)
        cv2.putText(img.data, ptext, pt, cv2.FONT_HERSHEY_SCRIPT_COMPLEX,2, [0, 255, 0], 2)
        cv2.imwrite(newimg,img.data)

    # Takes the actual classes (in answers) and the predictions made (in predictions) and calculates the F1 Score of the predictions.        
    
    def benchmark(self, answers, predictions):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return f1_score(answers, predictions, average="micro")
        except ValueError:
            return f1_score(answers,predictions, average="binary", pos_label = answers[0])