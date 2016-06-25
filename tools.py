import pandas as pd
import numpy as np
import image as im
import copy as c
import classifier as cf
import random, itertools, glob

class environment(object):
    
    # Takes a CSV file with image paths and their respective categories as input and outputs an environment object which contains the images along with 
    # a list of all categories that the images belong to.  See images.csv for an example of a CSV file used to initialize this object.
    
    def __init__(self,imagecsvs=None):
        if imagecsvs is not None:
            rawfile = pd.read_csv(imagecsvs)
            self.categories = list(set(rawfile['category'].tolist()))
            self.imgs = {}
            for i in self.categories:
                tmplist = []
                for j in rawfile[rawfile['category'] == i]['paths'].tolist():            
                    tmplist.append(im.image(j,i))
                self.imgs[i] = c.copy(tmplist)
                
    # Outputs a tuple of lists containing a training set of images (in the first element) and a testing set of images (in the second element).  The first parameter, 
    # trainsize, specifies the # of training images to randomly select from EVERY category.  If you set trainsize = 20 and you have 3 categories, your entire training
    # set will have 60 total images, for example. The second parameter, testsize, works like trainsize except it specifies the # of images that aren't selected for 
    # the training set to be put into the testing set from every category.
       
    def rand_train_test(self, trainsize=0,testsize=None):
        train = []
        test = []
        for i in self.categories:
            (t1,t2) = rand_split(self.imgs[i],trainsize=trainsize,testsize=testsize)
            train += t1
            test += t2
        return (train,test)
        
    # This is a helper function that calls rand_train_test() and creates several training-testing splits and puts them in a list. See the description for 
    # rand_train_test() for what trainsize and testsize mean.  The sets variable specifies the # of training-testing splits to make.  For example, if you want
    # 20 training images from each category, 10 testing images from each category, and 30 splits, you would run: test_sets(20, 10, 30)
    
    def test_sets(self,trainsize,testsize,sets):
        splits = []
        for i in range(0,sets):
            splits.append(self.rand_train_test(trainsize,testsize))
        return splits
    
    # Benchmarks a set of Classifier parameters. See rand_train_test() for what trainsize and testsize are for. See test_sets() for what sets is for. See 
    # classifier.py for the last 9 parameters which are classifier settings.    
    
        
    def benchmark(self,trainsize, testsize, sets, clf, feature_finder, convert_grey, rootsift, pca_before_kmeans, kclusters, pca_ratio, tfidf, incremental_threshold):
        mysplit = self.test_sets(trainsize,testsize,sets)
        traintime = []
        testtime = []
        f1 = []
        for (train,test) in mysplit:
            cf1 = cf.classifier(clf, feature_finder, convert_grey, rootsift, pca_before_kmeans, kclusters, pca_ratio, tfidf, incremental_threshold)
            cf1.incremental_train(train)
            f1.append(cf1.test(test))
            traintime.append(cf1.train_time)
            testtime.append(cf1.test_time)
        print "Average Training Time: {} seconds".format(np.average(traintime))
        print "Average Testing Time: {} seconds".format(np.average(testtime))
        print "Average F1 Score: {}".format(np.average(f1))
        



    # The following is a framework for doing a random gridsearch on a set of parameters and summarizing the results of the grid search 
    # in a Pandas dataframe in the results.  The first 3 parameters, trainsize, testsize, and sets are the same as in benchmark() above.
    # The "tries" parameter specifies how many random parameter combinations you want to test.  For example, if there are 1024 possible 
    # combinations but you only have time to test 100, then set tries = 100. The last 9 parameters correspond to the same Classifier 
    # parameters specified above but they must be inputted as a list where each list contains all possible values of a particular parameter 
    # you want to test.  See the iPython notebook for an example.

    
    def rand_grid_search(self,trainsize,testsize,sets,tries, Clf, Feature_finder,Convert_grey, Rootsift, Pca_before_kmeans, Kclusters, Pca_ratio, Tfidf, Incremental_threshold):
        allparms = [Clf, Feature_finder,Convert_grey, Rootsift, Pca_before_kmeans, Kclusters, Pca_ratio, Tfidf, Incremental_threshold]
        iterlist = itertools.product(*allparms)
        mysplit = self.test_sets(trainsize,testsize,sets)
        results = pd.DataFrame(columns=['a1','a2','a3','a4','a5','a6','a7','a8','a9','a10','a11','a12'])
        mylist = []
        for i in iterlist:
            mylist.append(i)
        for p in random.sample(mylist,tries):
            traintime = []
            testtime = []
            f1 = []
            for (train,test) in mysplit:
                cf1 = cf.classifier(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8])
                cf1.incremental_train(train)
                f1.append(cf1.test(test))
                traintime.append(cf1.train_time)
                testtime.append(cf1.test_time)
            avgtrain = np.average(traintime)
            avgtest = np.average(testtime)
            avgf1 = np.average(f1)
            results = results.append({'a1': avgf1, 'a2': avgtrain, 'a3': avgtest, 'a4': type(p[0]).__name__, 'a5': p[1], 'a6': p[2], 'a7': p[3], 'a8': p[4], 'a9': p[5], 'a10': p[6], 'a11': p[7], 'a12': p[8]},ignore_index = True)
        results = results.rename(columns={'a1': 'F1', 'a2': 'Avg Train Time', 'a3': 'Avg Test Time', 'a4': 'CLF', 'a5': 'Feature Finder','a6': 'Convert Grey', 'a7': 'RootSIFT','a8':'PCA First','a9': 'K Clusters','a10': 'PCA Ratio','a11': 'TF-IDF', 'a12': 'Incremental Threshold'})
        results = results.sort_values(by="F1",ascending=False)
        return results


    # This is an express function for quickly training a classifier on a set of images to be used immediately for image recognition.  It'll identify the image 
    # category in your set with the least samples and randomly select that many samples from each of your image categories to put together as a training set.
    # It'll then use the 9 parameteres you specified to train a classifier.  See classifier.py on what these parameters mean.
    
        
    def train_all(self, clf, feature_finder, convert_grey, rootsift, pca_before_kmeans, kclusters, pca_ratio, tfidf, incremental_threshold):
        n = 9999999999999
        for cat in self.categories:
            n = min(n,len(self.imgs[cat]))
        (train,test) = self.rand_train_test(n,0)
        c1 = cf.classifier(clf, feature_finder, convert_grey, rootsift, pca_before_kmeans, kclusters, pca_ratio, tfidf, incremental_threshold)
        c1.incremental_train(train)
        return c1


class caltech_environment(environment):
    
    
    # This child class makes it easier to load the images from the Caltech dataset easily. It assumes the directory that the image resides in is the image class.
    # Pass the list of directories you want to load images from in the dirlist variable when you create a caltech_environment object. For example, if you want 
    # the airplanes, gun, and dolphin classes, do: caltech_environment(dirlist=['airplanes','gun','dolphin']) .

    def __init__(self,rootdir="caltech",dirlist=None):
        self.categories = list(set(dirlist))
        self.imgs = {}
        for i in self.categories:
            tmplist = []
            for j in glob.glob(rootdir + "/" + i + "/*.jpg"):
                tmplist.append(im.image(j,i))
            self.imgs[i] = c.copy(tmplist)        
     

# This is a helper function for taking a list and splitting it into training and testing sets based on the trainsize and testsize variables.  It first selects
# a random number of elements from the list based on trainsize and puts it in the training set. Then it selects among the elements left over an amount equal 
# to testsize and puts it in the testing set.  And it returns the training set and testing set as a tuple.            
    
    
def rand_split(inputlist,trainsize=0,testsize=None):
    mylist = c.copy(inputlist)
    n = len(mylist)
    trainsize = min(trainsize,n)
    train = random.sample(mylist,trainsize)
    tmp = mylist
    for i in train:
        tmp.remove(i)
    if testsize is None:
        test = tmp
    else:
        testsize = min(testsize,n-trainsize)
        test = random.sample(tmp,testsize)
    return (train,test)