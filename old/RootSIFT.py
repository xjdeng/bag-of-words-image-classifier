# import the necessary packages
# Credit? See http://www.pyimagesearch.com/2015/04/13/implementing-rootsift-in-python-and-opencv/
import numpy as np
import cv2
 
class RootSIFT:
    def __init__(self):
        # initialize the SIFT feature extractor
        self.extractor = cv2.DescriptorExtractor_create("SIFT")
 
    def compute(self, image, kps, eps=1e-7):
        # compute SIFT descriptors
        (kps, descs) = self.extractor.compute(image, kps)
 
        # if there are no keypoints or descriptors, return an empty tuple
        if len(kps) == 0:
		return ([], None)
 
        # apply the Hellinger kernel by first L1-normalizing and taking the
        # square-root
        descs /= (descs.sum(axis=1, keepdims=True) + eps)
        descs = np.sqrt(descs)
	  #descs /= (np.linalg.norm(descs, axis=1, ord=2) + eps)
 
	  # return a tuple of the keypoints and descriptors
        return (kps, descs)
  
    def detectAndCompute(self, image):
        detector = cv2.FeatureDetector_create("SIFT")
        kps = detector.detect(image)
        extractor = cv2.DescriptorExtractor_create("SIFT")
        (kps, descs) = extractor.compute(image, kps)
        (kps, descs) = self.compute(image, kps)
        return (kps, descs)
    
    def orb_run(self, image):
        detector = cv2.FeatureDetector_create("ORB")
        kps = detector.detect(image)
        extractor = cv2.DescriptorExtractor_create("ORB")
        (kps, descs) = extractor.compute(image, kps)
        (kps, descs) = self.compute(image, kps)
        return (kps, descs)