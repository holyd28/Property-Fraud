import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

class AnomalyDetector:
    def __init__(self, neighbours = 5):
        # observe the 5 closest houses to check if property is an anomaly
        # use "ball_tree" algorithm which is optimised to find neighbours efficiently in spatial data
        self.knn = NearestNeighbors(n_neighbors=neighbours,algorithm="ball_tree")
        
    # to obtain the anomalous properties in the dataset
    def fit_predict(self,df):
        data = df.copy()
        coordinates = data[["lat","long"]].values # coordinates as a matrix (nested list)
        
        # training the model
        self.knn.fit(coordinates)
        
        # extract indices of 5 closest neighbours for each property
        distance, indices = self.knn.kneighbors(coordinates) 
        
        # calculating the average price of neighbours for each property
        neighbours_avg_price = []
        for i in range(len(data)):
            # get neighbours indices for property i, excluding itself (distance = 0)
            neighbour_indices = indices[i][1:]
            
            # find entry of each neighbour & get the average price
            avg_price = data.iloc[neighbour_indices]["price_per_sqft"].mean()
            
            neighbours_avg_price.append(avg_price)
            
        data["neighbours_avg_price"] = neighbours_avg_price
        
        # obtain ratio between the price of each property & its neighbours
        data["price_ratio"] = data["price_per_sqft"] / data["neighbours_avg_price"]
        
        # if house is sold for < 50% of the neighbours' average price, mark as anomaly
        data["is_anomaly"] = data["price_ratio"] < 0.50
        
        return data
