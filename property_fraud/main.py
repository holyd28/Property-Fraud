import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.anomaly import AnomalyDetector

def main():
    
    # load dataset
    try:
        print("Loading dataset...")
        # read file as a dataframe
        data = pd.read_csv("data/kc_house_data.csv")
        print("Dataset loaded successfully.")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # create our metric for comparison (price per square foot) to ensure fairness
    data["price_per_sqft"] = data["price"] / data["sqft_living"]
    
    print(f"Running KNN Anomaly Detection on {len(data)} properties...")
    
    # initialise detector 
    detector = AnomalyDetector(neighbours = 5)
    
    # obtain the updated dataframe with anomalies marked
    result = detector.fit_predict(data)
    
    # isolate on anomalous properties
    anomalies = result[result["is_anomaly"] == True]
    
    print(f"Anomalous properties detected: {len(anomalies)}")
    
    # save list of anomalies to results folder as a csv file
    anomalies.to_csv("results/property_anomalies.csv", index = False)
    print("List of anomalies saved to 'results/property_anomalies.csv'")
    
    print("Generating map...")
    
    # set size of image
    plt.figure(figsize=(10,10))
    
    # plotting all properties
    plt.scatter(result["long"], result["lat"], c = "grey", s = 5, alpha = 0.2, label = "Properties with market price")
    
    # overlaying anomalous properties
    plt.scatter(anomalies["long"], anomalies["lat"], c = "red", s = 20, alpha = 0.8, label = "Properties with suspicious prices")
    
    plt.title("Property Map with Anomalies Highlighted", fontsize = 16)
    plt.xlabel("Longitude", fontsize = 14)
    plt.ylabel("Latitude", fontsize = 14)
    plt.legend()
    plt.grid(True, alpha = 0.3)
    
    # save map to results folder
    plt.savefig("results/anomalous_property_map.png")
    print("Map saved to 'results/anomalous_property_map.png'")
    
if __name__ == "__main__":
    main()
