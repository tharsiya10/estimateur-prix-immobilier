import pandas as pd
import numpy as np
import preprocess
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import BallTree


def split_train_test(data, X_features, Y_features):
    region_columns = [col for col in data.columns if col.startswith('region_')]
    regions = data[region_columns].values
    train_x, test_x, train_y, test_y = train_test_split(
        X_features, Y_features, test_size=0.2, 
        random_state=42, stratify=regions
    )
    return train_x, test_x, train_y, test_y

def poly_linear_model(train_x, test_x, train_y) :
    
    poly = PolynomialFeatures(degree=2)
    train_x_poly = poly.fit_transform(train_x)
    test_x_poly = poly.transform(test_x) 
    model = LinearRegression()
    model.fit(train_x_poly, train_y)
    return model.predict(test_x_poly)

def random_forest_regressor(X, train_x, train_y) :
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    forest = RandomForestRegressor(n_estimators=30, min_samples_split=5)
    forest.fit(train_x, train_y)
    pred_y = forest.predict(X)
    print(pred_y)
    return pred_y

def evaluate(test_x, test_y, regression) :
    pred_y = regression(test_x)
    mae = mean_absolute_error(test_y, pred_y)
    print(f"Mean Absolute Error (MAE): {round(mae, 2)}")
    plt.figure(figsize=(12, 5))
    plt.ticklabel_format(style='plain')
    ax = sns.regplot(x=test_y, y=pred_y, line_kws={"color": "black"}, scatter_kws={'s': 8})
    ax.set(xlabel ="Actual Price", ylabel = "Predicted Price", title="Predicted Prices vs. Actual Prices")
    plt.show()

    residual = test_y - pred_y
    r2 = r2_score(test_y, pred_y)
    print(f"R-squared (R2): {round(r2, 2)}")

    ax = sns.displot(residual, kind='hist', height=4, aspect=3)
    ax.set(xlabel ="Residual", ylabel = "Density", title ='Density of the Price Residuals')

    plt.ticklabel_format(style='plain', axis='x')
    plt.show()

def get_my_bien(data, superficie, latitude, longitude, region, type_local) :
    model = BallTree(data[["latitude", "longitude"]].values, 
                         leaf_size=2, metric='haversine')
    k_neighbors =  (int((len(data * 5) // 100))) + 1
    my_bien_lat_long = np.array([[latitude, longitude]])
    dist, indices = model.query(my_bien_lat_long, 
                                    k=k_neighbors)
    dist_moy = np.mean(dist[:,1:], 1)
    print(dist)
    print(dist_moy)
    print(indices[0])
    prix_par_m_carre_neighbors = 0
    for i in indices[0] :
        prix_par_m_carre_neighbors += data.at[i, "prix_par_m_carre"]
    prix_par_m_carre_neighbors /=  k_neighbors
    prix_moy_quartier = prix_par_m_carre_neighbors
    
    my_bien = pd.DataFrame()

    my_bien.at[0, "surface_reelle_bati"] = superficie
    my_bien["longitude"] = longitude
    my_bien["latitude"] = latitude
    my_bien = preprocess.add_tranche_m_carre(my_bien, 20)
    regions = [col for col in data.columns if col.startswith('region_')]
    types = [col for col in data.columns if col.startswith('type_local_')]
    for r in regions :
        if r != region :
            my_bien[r] = 0
        else :
            my_bien[r] = 1
    for t in types :
        if t != type_local :
            my_bien[t] = 0
        else :
            my_bien[t] = 1
    my_bien["distance_moy"] = dist_moy
    my_bien["prix_moy_quartier"] = prix_moy_quartier
    print(my_bien["surface_reelle_bati"])
    print(my_bien["longitude"])
    print(my_bien["latitude"])
    return my_bien


def price_predict(my_bien, train_x, train_y) :
    my_price = poly_linear_model(train_x, my_bien, train_y)
    return my_price
