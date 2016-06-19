import cv2

class image(object):
    
    # The image object takes the path of an image and an assigned category and outputs an image object which contains a numpy object containing the image pixel data 
    # in the data member variable and the category in the category member variable.
    
    def __init__(self, path, category):
        self.data = cv2.imread(path, cv2.CV_LOAD_IMAGE_COLOR)
        self.category = category
        if self.data is None:
            print "Warning! {} did not load!".format(path)