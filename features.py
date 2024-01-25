from statistics import correlation
import preprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree
import seaborn as sns
from scipy.stats import spearmanr

def separate_types(biens) :
    appart = biens[(biens["type_local_Appartement"] == 1) & (biens["nature_mutation_Vente"] == 1)]
    appart.reset_index(drop=True, inplace=True)
    
    maison = biens[(biens["type_local_Maison"] == 1) & (biens["nature_mutation_Vente"] == 1)]
    maison = maison[(maison["prix_par_m_carre"] > 100) & (maison["surface_reelle_bati"] > 9) ]
    maison.reset_index(drop=True, inplace=True)

    local = biens[(biens["type_local_Local industriel. commercial ou assimilé"] == 1) & (biens["nature_mutation_Vente"] == 1)]
    local.reset_index(drop=True, inplace=True)

    return appart, maison, local

def separate_by_tranche_mean_neighborhood(biens_type) :
    tranches = biens_type["tranche_par_m_carre"].unique()
    print(tranches)
    biens_tranche = [None] * len(tranches)
    for b in range(len(biens_tranche)) :
        biens_tranche[b] = biens_type[biens_type["tranche_par_m_carre"] == tranches[b]]
        biens_tranche[b] = pd.concat(mean_neighborhood(biens_tranche[b]), ignore_index=True)
    return biens_tranche


def mean_neighborhood(biens_type) :
    biens_type["distance_moy"] = np.zeros(len(biens_type))
    regions = [col for col in biens_type.columns if col.startswith('region_')]
    data_regions = [None] * len(regions)
    for k in range(len(regions)) :
        data_regions[k] = biens_type[biens_type[regions[k]] == 1]
        data_regions[k] = data_regions[k].reset_index(drop=True)
        if(len(data_regions[k]) == 0) : 
            continue
        model = BallTree(data_regions[k][["latitude", "longitude"]].values, 
                         leaf_size=2, metric='haversine')
        k_neighbors =  (int((len(data_regions[k] * 5) // 100))) + 1

        dist, indices = model.query(data_regions[k][["latitude", "longitude"]].values, 
                                    k=k_neighbors)
        # exclude itself
        data_regions[k]["distance_moy"] = np.mean(dist[:,1:], 1)

        prix_m_carre = pd.DataFrame()
        prix_m_carre["prix_par_m_carre"] = np.zeros(len(data_regions[k]))
        # exclude itself
        for i in range(1, k_neighbors) :
            prix_m_carre += pd.DataFrame(
                data_regions[k].iloc[indices[:,i],:]["prix_par_m_carre"]).reset_index(drop=True)
        prix_m_carre = prix_m_carre / k_neighbors
        
        data_regions[k]["prix_moy_quartier"] = prix_m_carre.values
        data_regions[k]['prix_moy_quartier'] = data_regions[k]['prix_moy_quartier'].fillna(
            data_regions[k]['prix_par_m_carre'])
        data_regions[k]['distance_moy'] = data_regions[k]['distance_moy'].fillna(0)
        print("======"+regions[k]+"======")
        print(data_regions[k])

    return data_regions


def corrman(biens) :
    X_features = biens.drop(["valeur_fonciere",
                "prix_par_m_carre",
                "nature_mutation_Vente",
                "nombre_lots",
                "surface_terrain"], axis=1)
    Y_features = biens["prix_par_m_carre"]
                
    X_features_columns = np.array(X_features.columns)
    feature_df = pd.DataFrame()
    correlation = []
    p_value = []
    for feature in X_features_columns :
        corr, pval = spearmanr(biens[feature], biens["prix_par_m_carre"])
        correlation.append(round(corr, 2))
        p_value.append(round(pval, 4))
    feature_df["feature"] = X_features_columns
    feature_df["spearman_correlation"] = correlation
    feature_df["spearman_p_value"] = p_value
    print(feature_df)

    # plt.figure(figsize = (16, 8))
    # sns.heatmap(spearmancorr, xticklabels=spearmancorr.columns, yticklabels=spearmancorr.columns, annot=True)
    # plt.title('Spearman Correlation Matrix', fontsize=14, fontweight='bold')
    # plt.show()
    return X_features, Y_features

def draw_graph(data, x, y):
    sns.lmplot(x='prix_moy_quartier', y='prix_par_m_carre', hue='tranche_par_m_carre', data=data, aspect=2)
    plt.xlabel("Prix moyen du quartier")
    plt.ylabel("Prix par m^2")
    plt.show()