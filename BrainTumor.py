import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from image_features import image_features

dir = './brain_tumor_dataset/yes/'
dir2 = './brain_tumor_dataset/no/'

def KNN(x,y, n):
    x = pd.get_dummies(data = x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2) 
    model = KNeighborsClassifier(n_neighbors=n)
    model.fit(x_train,y_train)
    predict = model.predict(x_test)
    return metrics.accuracy_score(predict, y_test)*100

def FeatureExtraction(dir, dir2):
    images = []
    for filename in os.listdir(dir):
        images.append(dir+filename)    
        tumor.append(1)


    for filename in os.listdir(dir2):
        images.append(dir2+filename)
        tumor.append(0)

    features = image_features(images)
    features = pd.DataFrame(features)
    features['tumor'] = tumor
    features.to_csv('./BrainTumorDataset.csv', index  = False, header = False)    

if __name__ == '__main__':

    # FeatureExtraction(dir, dir2)

    tumor=pd.read_csv('./BrainTumorDataset.csv')
    x = tumor.iloc[: , 0: -1]
    y = tumor.iloc[: , -1]

    print("knn accuracy:",KNN(x, y, 1),'%')
