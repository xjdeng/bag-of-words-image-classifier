import cv2, time
import numpy as np
import classifier as c

FLANN_INDEX_KDTREE = 0


class flannclf(object):
    
    def __init__(self):
        self.flann = {}
        self.index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        self.search_params = dict(checks=50)   # or pass empty dictionary
        self.h = None
    
    def fit(self,x,y):
        n = min(len(x),len(y))
        for i in range(0,n):
            mystr = str(y[i])
            if mystr not in self.flann:
                self.flann[mystr] = cv2.FlannBasedMatcher(self.index_params,self.search_params)
            self.flann[mystr].add(x[i])

    
    def partial_fit(self,x,y):
        self.fit(x,y)

    def predict(self,Z):
        answers = []
        for z in Z:
            answers.append(self.predict_one(z))
        return np.array(answers)
    
    def predict_one(self,z):
        categories = self.flann.keys()
        bestc = categories[0]
        bestm = 0
        for ca in categories:
            n_matches = 0
            matches = self.flann[ca].knnMatch(z,k=2)
            for (m1,m2) in matches:
                if m1.distance < 0.7*m2.distance:
                    n_matches += 1
            if n_matches > bestm:
                bestc = ca
                bestm = n_matches
        return bestc
    
class flann_classifier(c.classifier):

    def __init__(self, fclf, feature_finder="ORB",  convert_grey=True, rootsift=True, pca_ratio = 0.5):
        super(flann_classifier, self).__init__(clf=fclf, feature_finder = feature_finder, convert_grey=convert_grey, rootsift = rootsift, pca_ratio = pca_ratio)
     
    def train(self, imglist):
        self.train_time = time.time()
        self.getDescriptors(imglist)
        self.clf.fit(self.des_list, self.image_classes)
        self.train_time = time.time() - self.train_time
    
    def test(self, test_cases):
        self.test_time = time.time()
        answer_list = []
        for i in test_cases:
            answer_list.append(i.category)
        answers = np.array(answer_list)
        self.getDescriptors(test_cases)
        predictions = self.predict()
        self.test_time = time.time() - self.test_time
        return self.benchmark(answers=answers, predictions=predictions)   

    def predict(self):
        return self.clf.predict(self.des_list)