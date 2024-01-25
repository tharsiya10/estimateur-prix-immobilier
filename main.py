import preprocess
import features
import all_models
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def test_separation_type(biens) :
    appart, maison, local = features.separate_types(biens)
    appart = features.mean_neighborhood(appart)
    maison = features.mean_neighborhood(maison)
    local = features.mean_neighborhood(local)
    appart_regions = pd.concat(appart, ignore_index=True)
    maison_regions = pd.concat(maison, ignore_index=True)
    local_regions = pd.concat(local, ignore_index=True)
    biens_regions = pd.concat([maison_regions, appart_regions, local_regions], ignore_index=True)
    return biens_regions

def test_separation_type_tranche(biens) :
    appart, maison, local = features.separate_types(biens)
    appart = features.separate_by_tranche_mean_neighborhood(appart)
    maison = features.separate_by_tranche_mean_neighborhood(maison)
    local = features.separate_by_tranche_mean_neighborhood(local)
    
    appart_regions = pd.concat(appart, ignore_index=True)
    maison_regions = pd.concat(maison, ignore_index=True)
    local_regions = pd.concat(local, ignore_index=True)
    biens_regions = pd.concat([maison_regions, appart_regions, local_regions], ignore_index=True)
    return biens_regions

def test_separation_tranche(biens) :
    biens = features.separate_by_tranche_mean_neighborhood(biens)
    biens_regions = pd.concat(biens, ignore_index=True)
    return biens_regions

def test_no_separation(biens) :
    bien_no_sep = features.mean_neighborhood(biens)
    biens_regions = pd.concat(bien_no_sep, ignore_index=True)
    return biens_regions

if __name__ == "__main__" :
    biens_2023 = preprocess.cleanDataset("2023.csv")
    # biens_2022 = preprocess.cleanDataset("2022.csv")
    # biens_2021 = preprocess.cleanDataset("2021.csv")
    # biens_2020 = preprocess.cleanDataset("2020.csv")
    # biens = pd.concat([biens_2023, biens_2022, biens_2021, biens_2020], ignore_index=True)
    biens = biens_2023
    print(preprocess.pivot_table_type_nature(biens))
    print(biens)
    features_data = ["prix_par_m_carre", "tranche_par_m_carre", "type_local_Maison", "type_local_Appartement", "type_local_Local industriel. commercial ou assimilé", "prix_moy_quartier"]

    """TEST SANS SEPARATION"""
    biens_regions = test_no_separation(biens)
    """TEST AVEC SEPARATION DES TYPES"""
    # biens_regions = test_separation_type(biens)
    """SEPARATION DES BIENS SIMILAIRES EN TERMES DE M_CARRE"""
    # biens_regions = test_separation_tranche(biens)
    """SEPARATION EN TYPE DE BIENS ET EN TERMES DE M_CARRE"""
    # biens_regions = test_separation_type_tranche(biens)
    
    spearmancorr = biens_regions.corr(method='spearman')
    X_features, Y_features = features.corrman(biens_regions)
    train_x, test_x, train_y, test_y = all_models.split_train_test(biens_regions, X_features, Y_features)
    print("==TRAIN X==")
    print(train_x)
    print("==TEST_X==")
    print(test_x)
    print("===TRAIN Y===")
    print(train_y)
    print("===TEST Y===")
    print(test_y)

    """POLYNOMIAL FEATURES & LINEAR REGRESSION"""
    all_models.evaluate(test_x, test_y, lambda x : all_models.poly_linear_model(train_x, x, train_y))
    # my_bien = all_models.get_my_bien(biens_regions, superficie=46, latitude=48.848713, longitude=2.506329, region="region_IDF", type_local="type_local_Appartement")
    # my_price = all_models.price_predict(my_bien, train_x, train_y)
    # print("===PREDICTED PRICE===")
    # print(my_price)
    
    """RANDOM FOREST
    """
    # all_models.evaluate(test_x, test_y, lambda x : all_models.random_forest_regressor(x, train_x, train_y))
    
    
