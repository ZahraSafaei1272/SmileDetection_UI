import os
import re
import cv2
import glob
import random
import joblib
import numpy as np
from sklearn.svm import SVC
from natsort import natsorted
from sklearn import metrics
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from skimage import feature, color, transform, util
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def lbp_feature_vector(image, radius=3):
    # Convert to grayscale if needed
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    #gray = cv2.resize(gray, (128,128))
    n_points = 8 * radius

    # Compute LBP (uint8)
    lbp = feature.local_binary_pattern(gray, n_points, radius, method="uniform")

    hist, _ = np.histogram(lbp.ravel(), 256,(0,256))
    # Normalize
    #hist = hist.astype("float")
    #hist /= (hist.sum() + 1e-7) 

    return hist




def hog_feature_vector(image, orientations=8, pixels_per_cell=(16,16), cells_per_block=(1,1)):
    # Convert to grayscale if needed
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    #gray = cv2.resize(gray, (128,128))
    hog_features, hog_image = hog(
        gray, 
        orientations, 
        pixels_per_cell, 
        cells_per_block, 
        block_norm='L2-Hys',
        visualize=True 
    )
    #print("HOG feature vector length:", hog_features.shape)

    # Show visualization
    #plt.imshow(hog_image, cmap="gray")
    #plt.title("HOG Visualization")
    #plt.axis("off")
    #plt.show()
    return hog_features



def load_images(image_paths):
    images = [cv2.imread(path) for path in image_paths]
    return images



def extract_feature(images):
    l_features = [lbp_feature_vector(img) for img in images]
    lbp_feature = np.array(l_features)
    #print("Shape:", lbp_feature.shape)
    h_features = [hog_feature_vector(img) for img in images]
    hog_feature = np.array(h_features)
    #print("Shape:", hog_feature.shape)
    X = np.hstack((hog_feature, lbp_feature))
    #print(X.shape)
    return X




def prepare_label(filename, r):
    y = np.loadtxt(filename)
    y = y[:,0]
    y = np.repeat(y, r)
    #print(len(y))
    return y




def prepare_data(X, y, scaler_name = 'scaler.joblib'):
    #train_test split
    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=.33, shuffle=True, random_state=123, stratify=y)
    #scaling data
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_std_train = scaler.fit_transform(X_train)
    joblib.dump(scaler, scaler_name)
    X_std_test = scaler.transform(X_test)
    #scaler = StandardScaler()
    #X_std_train = scaler.fit_transform(X_train)
    #X_std_test = scaler.fit_transform(X_test)
    
    ###dimension reduction
    #pca = PCA(n_components=100) 
    #X_train_pca = pca.fit_transform(X_train)
    #X_test_pca = pca.transform(X_test)
    
    return X_std_train, X_std_test, y_train, y_test




def train(X_train, y_train, model_name = 'svm_model.joblib'):
    svc = SVC(C=1, kernel='poly', class_weight="balanced")
    svc.fit(X_train, y_train)
    y_pred_train = svc.predict(X_train)
    accuracy = np.mean(y_train == y_pred_train)
    print("Train accuracy:", accuracy)
    # Save the trained classifier using joblib
    joblib.dump(svc, model_name)



def test(X_test, y_test, model_name = 'svm_model.joblib'):
    # Load SVM model from disk
    model = joblib.load(model_name)
    # Make predictions on the testing set using the loaded model
    y_pred = model.predict(X_test)
    accuracy_test = np.mean(y_test == y_pred)
    print("Test accuracy:", accuracy_test)
    print(metrics.classification_report(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))





def main():
    image_paths_new = natsorted(glob.glob('aug_images/*.jpg'))
    images = load_images(image_paths_new)
    X = extract_feature(images)
    y = prepare_label("labels.txt", 5)
    X_train, X_test, y_train, y_test = prepare_data(X, y)
    train(X_train, y_train, model_name = 'svm_model.joblib')
    test(X_test, y_test, model_name = 'svm_model.joblib')




if __name__ == '__main__':
    main()






