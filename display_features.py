import cv2
import numpy as np

def get_features(imgpath, algorithm="SIFT"):
    img = cv2.imread(imgpath)
    return get_features_from_img(img,algorithm)

def get_features_from_img(img, algorithm="SIFT"):
    if img is None:
        print "Not a valid image!"
        return None
    detector = cv2.FeatureDetector_create(algorithm)
    kps = detector.detect(img)
    img2 = cv2.drawKeypoints(img, kps)
    return img2    

def avg_features(imglist, algorithm="SIFT"):
    detector = cv2.FeatureDetector_create(algorithm)
    extractor = cv2.DescriptorExtractor_create(algorithm)
    keypoints = []
    for i in imglist:
        grey = cv2.cvtColor(i.data, cv2.COLOR_BGR2GRAY)
        kps = detector.detect(grey)
        (kps, descs) = extractor.compute(grey, kps)
        if descs is None:
            keypoints.append(0)
        else:
            keypoints.append(len(descs))
    return np.mean(keypoints)
        

def display_image(img):
    cv2.imshow("Features",img)
    
def save_img(filename, img):
    cv2.imwrite(filename, img)