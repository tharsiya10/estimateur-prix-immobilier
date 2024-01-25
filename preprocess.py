import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import BallTree

def cleanDataset(year) :
    column_data_types = {"nature_mutation": str,"valeur_fonciere": float,
                         "adresse_numero": str,"adresse_nom_voie": str,
                        "code_postal": str,"type_local": str,"nombre_lots": float,
                        "surface_reelle_bati": float,
                        "surface_terrain": float,"longitude": float,
                        "latitude": float,"code_departement": str, 
    }
    columns_to_load = ["nature_mutation","valeur_fonciere",
                       "adresse_numero","adresse_nom_voie",
                        "code_postal","type_local",
                        "nombre_lots","surface_reelle_bati",
                        "surface_terrain","longitude",
                        "latitude","code_departement",
    ]
    biens = pd.read_csv(
        year, sep=",", usecols=columns_to_load, dtype=column_data_types
    )
    # keep only Vente and VEFA
    biens = biens[
        (biens["nature_mutation"] == "Vente")
        | (biens["nature_mutation"] == "Vente en l'état futur d'achevement")
    ]
    # keep only nombre lots == 0 and nb lots == 1
    biens = biens[(biens["nombre_lots"] == 0) | (biens["nombre_lots"] == 1)]

    # keep bien with not zero surface reelle bati and valeur fonciere
    biens = biens[biens["surface_reelle_bati"] != 0]
    biens = biens[biens["valeur_fonciere"] != 0]

    biens = biens[biens["surface_reelle_bati"].notna()]
    biens = biens[biens["valeur_fonciere"].notna()]

    biens = biens[biens["type_local"].notna()]
    biens = biens[biens["nature_mutation"].notna()]
    biens = biens[biens["code_departement"].notna()]
    biens = biens[biens["adresse_nom_voie"].notna()]
    biens = biens[biens["code_postal"].notna()]
    biens = biens[biens["adresse_numero"].notna()]

    biens["region"] = biens["code_departement"].apply(getRegion)
    biens["region"] = biens["region"].astype(str)
    biens = biens[biens["region"].notna()]
    biens = biens[biens["region"] != "None"]

    biens = biens[biens["latitude"].notna()]
    biens = biens[biens["longitude"].notna()]
    biens = remove_duplicate(biens)

    # donnees necessaire pour Ball Tree
    biens["surface_terrain"] = biens["surface_terrain"].fillna(0)
    biens["valeur_fonciere"] = biens["valeur_fonciere"].apply(lambda x : round(x))
    biens["prix_par_m_carre"] = biens["valeur_fonciere"] / biens["surface_reelle_bati"]
    biens["prix_par_m_carre"] = biens["prix_par_m_carre"].apply(lambda x : round(x))
    biens = biens[biens["prix_par_m_carre"] < 10000.0]
    biens = biens[biens["surface_reelle_bati"] < 500.0]
    
    biens = biens.sort_values(by = ["region", "latitude", "longitude"])
    biens = biens[biens["type_local"] != "Dépendance"]

    biens = add_tranche_m_carre(biens, 20)
    biens = drop_columns_addr(biens)

    biens = one_hot_encoding_nature_mutation(biens)
    type_local = [col for col in biens.columns if col.startswith('type_local_')]
    print(type_local)
    biens.reset_index(drop=True, inplace=True)
    return biens

def one_hot_encoding_nature_mutation(biens) :
    columns_to_encode = ["region", "nature_mutation", "type_local"]
    df_to_encode = biens[columns_to_encode]
    df_encoded = pd.get_dummies(
        df_to_encode, columns=columns_to_encode, prefix=columns_to_encode
    )
    df_encoded = df_encoded.astype(int)
    biens = biens.join(df_encoded)
    biens = biens.drop(columns=["region", "nature_mutation", "type_local"])

    return biens

def drop_columns_addr(biens) :
    biens = biens.drop(columns=["adresse_nom_voie", "code_postal", "adresse_numero", "code_departement"])
    return biens

def pivot_table_type_nature(biens) :
    pivot = pd.pivot_table(biens, values="valeur_fonciere", index=["type_local_Maison", "type_local_Appartement", "type_local_Local industriel. commercial ou assimilé"],
                           columns=["nature_mutation_Vente"], aggfunc=lambda x: len(x))
    return pivot

# remove doublons
def remove_duplicate(data) :
    data = data.drop_duplicates(subset=['latitude', 'longitude'], keep='first')
    return data

# to have little fragment for testing faster
def extractRegion(data, region):
    data = data[data[region] == 1]
    # data.to_csv(region + ".csv", sep=",", encoding="utf-8")
    return data

def extractSameTypeLocal(data, type_local) :
    data = data[data[type_local] == 1]
    return data

def add_tranche_m_carre(data, tranche) :
    data["tranche_par_m_carre"] = tranche * (data["surface_reelle_bati"] // tranche)
    return data

def getRegion(zipcode):
    """Add the region to each line in the CSV file given the zipcode
    Args :
        zipcode in int
    Returns :
        the region according in str
    """
    dict_region_zip = {
        "Auvergne-Rhone-Alpes": [
            "1","01","3","03","7","07","15","26","38","42","43","63","69","73","74",],
        "Bourgogne-Franche-Comte": ["21", "25", "39", "58", "70", "71", "89", "90"],  #
        "Bretagne": ["22", "29", "35", "56"],
        "Centre-Val de Loire": ["18", "28", "36", "37", "41", "45"],
        "Corse": ["2A", "2B"],
        "Grand Est": ["8","08","10","51","52","54","55","57","67","68","88",
        ],
        "Hauts-de-France": ["2", "02", "59", "60", "62", "80"],
        "IDF": ["75", "77", "78", "91", "92", "93", "94", "95"],
        "Normandie": ["14", "27", "50", "61", "76"],
        "Nouvelle-Aquitaine": ["16","17","19","23","24","33","40","47","64","79","86","87",
        ],
        "Occitanie": ["9","09","11",
            "12",
            "30",
            "31",
            "32",
            "34",
            "46",
            "48",
            "65",
            "66",
            "81",
            "82",
        ],
        "Pays de la Loire": ["44", "49", "53", "72", "85"],
        "PACA": [
            "4",
            "04",
            "5",
            "05",
            "06",
            "6",
            "13",
            "83",
            "84",
        ],
    }
    for region in dict_region_zip.keys():
        if zipcode in dict_region_zip[region]:
            return region


def visualize_repartition_m_carre(biens) :
    tranches = biens["tranche_par_m_carre"].unique()
    plt.figure(figsize=(15,5))
    for idx, part in enumerate(tranches) :
        sns.countplot(data=biens[biens["tranche_par_m_carre"] == part], x="tranche_par_m_carre")
    plt.subplots_adjust(wspace=10.0)
    plt.xticks(rotation=90)
    plt.show()

def extractDepartement(data, dep) :
    data = data[data["code_departement"] == dep]
    return data


# fill surface reelle batie 
def fill_surface_relle_batie_med(data) :
    departements = data['code_departement'].unique()
    data_filled = [None] * len(departements)
    for i in range(len(departements)) :
        data_filled[i] = extractDepartement(data, departements[i])
        median_surface = data_filled[i]['surface_reelle_bati'].median()
        data_filled[i]['surface_reelle_bati'].fillna(median_surface, inplace=True)
    return pd.concat(data_filled, ignore_index=True)

# find k neighbors and med of surface
def fill_missing_surface_reelle_batie_neighbors(row, data) :
    if np.isnan(row["surface_reelle_bati"]) :
        data_departement = extractDepartement(data, row["code_departement"])
        features = data_departement[['latitude', 'longitude', 'surface_reelle_batie']].dropna()
        tree = BallTree(
            (features[["latitude", "longitude"]].values), leaf_size=2, metric="haversine"
        )
        _, indices = tree.query([[row["latitude"], row["longitude"]]], k=10)
        indices = indices[0][1:]
        median_surface_neighbors = np.median(features.iloc[indices]['surface_reelle_batie'])
        print(median_surface_neighbors)
        return median_surface_neighbors
    return row["surface_relle_bati"]

def fill_surface_reelle_batie_neighbors(data) :
    data["surface_reelle_bati"] = data.apply(fill_missing_surface_reelle_batie_neighbors, axis=1)
    return data

# fill missing valeur fonciere with med
def fill_valeur_fonciere_med(data) :
    departements = data['code_departement'].unique()
    data_filled = [None] * len(departements)
    for i in range(len(departements)) :
        data_filled[i] = extractDepartement(data, departements[i])
        median_surface = data_filled[i]['valeur_fonciere'].median()
        data_filled[i]['valeur_fonciere'].fillna(median_surface, inplace=True)
    return pd.concat(data_filled, ignore_index=True)

def fill_missing_valeur_fonciere(row, data) :
    if np.isnan(row["valeur_fonciere"]) :
        data_departement = extractDepartement(data, row["code_departement"])
        features = data_departement[["tranche_par_m_carre", "valeur_fonciere"]].dropna()
        median_val_fonciere = features["valeur_fonciere"].median()
        return median_val_fonciere
    else :
        return row["valeur_fonciere"]
    
def fill_valeur_fonciere_par_tranche(data) :
    data["valeur_fonciere"] = data.apply(fill_missing_valeur_fonciere, axis=1)
    return data
