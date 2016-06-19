import random, copy, cv2
import pandas as pd
import classifier

classmap = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']

myDataDir = "/Users/xjdeng/Downloads/imgs/train"

from sklearn.svm import SVC

tmpclf = SVC(C=100, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape=None, degree=3, gamma=0.0001, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)

def rand_split(inputlist,trainsize=0,testsize=None):
    mylist = copy.copy(inputlist)
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

def everybody():
    return ['p022', 'p049', 'p021', 'p026', 'p002', 'p041', 'p042', 'p045', 'p047', 'p024', 'p072', 'p075', 'p016', 'p015', 'p014', 'p039', 'p012', 'p035', 'p052', 'p051', 'p050', 'p056', 'p066', 'p064', 'p061', 'p081']


def load_images(mypd, dataDir=myDataDir):
    images = []
    for i in range(0,len(mypd)):
        mypath = dataDir + "/" + mypd['classname'].iloc[i] + "/" + mypd['img'].iloc[i]
        tmp = cv2.imread(mypath)
        images.append(cv2.cvtColor(tmp,cv2.COLOR_BGR2GRAY))
    return images

def filter_and_sample(mylist, ppl_list, samplesize):
    (imgs, classes) = ([],[])
    for i in ppl_list:
        tmppd = mylist[mylist['subject'] == i].sample(samplesize)
        imgs += load_images(tmppd)
        classes += tmppd['classname'].tolist()
    return (imgs,classes)
        

def run_with_parms(n_train_ppl,n_test_ppl,trainsize,testsize):
    mylist = pd.read_csv("driver_imgs_list.csv")
    (train_p,test_p) = rand_split(everybody(),n_train_ppl,n_test_ppl)
    (train_imgs, train_classes) = filter_and_sample(mylist, train_p, trainsize)
    (test_imgs, test_classes) = filter_and_sample(mylist, test_p, testsize)
    classifier1 = classifier.classifier(classmap)
    classifier1.prefit(train_imgs)
    classifier1.doPCA()
    classifier1.mainfit(classes=train_classes)
    classifier1.predict(test_imgs)
    print classifier1.benchmark(test_classes)
    return (classifier1, train_imgs, train_classes, test_imgs, test_classes)
    
def run_with_dt(n_train_ppl,n_test_ppl,trainsize,testsize):
    mylist = pd.read_csv("driver_imgs_list.csv")
    (train_p,test_p) = rand_split(everybody(),n_train_ppl,n_test_ppl)
    (train_imgs, train_classes) = filter_and_sample(mylist, train_p, trainsize)
    (test_imgs, test_classes) = filter_and_sample(mylist, test_p, testsize)
    from sklearn.tree import DecisionTreeClassifier
    classifier1 = classifier.classifier(classmap,clf=DecisionTreeClassifier())
    classifier1.prefit(train_imgs)
    classifier1.doPCA()
    classifier1.mainfit(classes=train_classes)
    classifier1.predict(test_imgs)
    print classifier1.benchmark(test_classes)
    return (classifier1, train_imgs, train_classes, test_imgs, test_classes)

def run_with_nb(n_train_ppl,n_test_ppl,trainsize,testsize):
    mylist = pd.read_csv("driver_imgs_list.csv")
    (train_p,test_p) = rand_split(everybody(),n_train_ppl,n_test_ppl)
    (train_imgs, train_classes) = filter_and_sample(mylist, train_p, trainsize)
    (test_imgs, test_classes) = filter_and_sample(mylist, test_p, testsize)
    from sklearn.naive_bayes import GaussianNB
    classifier1 = classifier.classifier(classmap,clf=GaussianNB())
    classifier1.prefit(train_imgs)
    classifier1.doPCA()
    classifier1.mainfit(classes=train_classes)
    classifier1.predict(test_imgs)
    print classifier1.benchmark(test_classes)
    return (classifier1, train_imgs, train_classes, test_imgs, test_classes)

def run_with_keras(n_train_ppl,n_test_ppl,trainsize,testsize):
    mylist = pd.read_csv("driver_imgs_list.csv")
    (train_p,test_p) = rand_split(everybody(),n_train_ppl,n_test_ppl)
    (train_imgs, train_classes) = filter_and_sample(mylist, train_p, trainsize)
    (test_imgs, test_classes) = filter_and_sample(mylist, test_p, testsize)
    from keras.models import Sequential
    from keras.layers import Dense
    classifier1 = classifier.classifier(classmap)
    classifier1.prefit(train_imgs)
    classifier1.doPCA()
    mymodel = Sequential()
    mymodel.add(Dense(12, input_dim = classifier1.PCA_components, init= 'uniform' , activation='relu' ))
    mymodel.add(Dense(8, init= 'uniform' , activation='relu' ))
    mymodel.add(Dense(1, init='uniform', activation='sigmoid'))
    mymodel.compile(loss= 'binary_crossentropy'  , optimizer= 'adam' , metrics=[ 'accuracy' ])
    classifier1.clf = mymodel
    classifier1.keras_fit(classes=train_classes)
    print classifier1.keras_benchmark(test_imgs,test_classes)
    return (classifier1, train_imgs, train_classes, test_imgs, test_classes)

def run_with_clf(myclf,n_train_ppl,n_test_ppl,trainsize,testsize):
    mylist = pd.read_csv("driver_imgs_list.csv")
    (train_p,test_p) = rand_split(everybody(),n_train_ppl,n_test_ppl)
    (train_imgs, train_classes) = filter_and_sample(mylist, train_p, trainsize)
    (test_imgs, test_classes) = filter_and_sample(mylist, test_p, testsize)
    classifier1 = classifier.classifier(classmap,clf=myclf)
    classifier1.prefit(train_imgs)
    classifier1.doPCA()
    classifier1.mainfit(classes=train_classes)
    classifier1.predict(test_imgs)
    print classifier1.benchmark(test_classes)
    return (classifier1, train_imgs, train_classes, test_imgs, test_classes)   



def grid_with_parms(n_train_ppl,n_test_ppl,trainsize,testsize):
    mylist = pd.read_csv("driver_imgs_list.csv")
    (train_p,test_p) = rand_split(everybody(),n_train_ppl,n_test_ppl)
    (train_imgs, train_classes) = filter_and_sample(mylist, train_p, trainsize)
    (test_imgs, test_classes) = filter_and_sample(mylist, test_p, testsize)
    from sklearn.grid_search import GridSearchCV
    from sklearn.metrics import make_scorer, f1_score
    getscore = make_scorer(f1_score,average="micro")
    param_grid = {'gamma': [0.0001, 1e-5, 1e-6, 1e-7]}
    myclf = GridSearchCV(SVC(C=100,class_weight="balanced"), param_grid,getscore)
    classifier1 = classifier.classifier(classmap,clf=myclf)
    classifier1.prefit(train_imgs)
    classifier1.doPCA()
    classifier1.mainfit(classes=train_classes)
    classifier1.predict(test_imgs)
    print classifier1.benchmark(test_classes)
    return (classifier1, train_imgs, train_classes, test_imgs, test_classes)