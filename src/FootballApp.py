from importlib import reload
import sys
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import StringType
from pyspark.sql.window import Window
from pyspark.sql.types import BooleanType
from pyspark.sql.types import IntegerType

spark = SparkSession.builder.getOrCreate()

def main(argv):
    #Récuperer le fichier à l'aide d'une fonction et affichage
    dframe = import_du_csv_match()
    
    #Renomme les colonnes X4 et X6 à l'aide d'une fonction et affichage
    dframe = renommer_les_colonnes(dframe)
    
    #Sélectionne les colonnes nécessaires à l'aide d'une fonction et affichage
    dframe = selection_des_colonnes(dframe)
    
    #Compléter les valeurs nulles dans les colonnes penalty par des 0 à l'aide d'une fonction et affichage
    dframe = remplacer_les_valeurs_vulles(dframe)
    
    #Filtrer et garder uniquement les matchs datant de mars 1980 à aujourd’hui
    dframe = filtrer_match_recent(dframe)
    
    dframe.show()
    
    # Création colonne boolean indiquant pour chaque ligne si le match a été joué à domicile (true) ou pas (false)
    dframe_a_domicile = a_domicile(dframe)
    dframe_a_domicile.show()
    
    
    #Appel à la fonction d'aggrégation
    dframe_statistiques = statistiques(dframe_a_domicile)
    dframe_statistiques.show()
    
#Fonction pour lire le fichier csv dans une dataframe    
def import_du_csv_match():
    
    df = spark.read.csv("src/df_matches.csv", header=True)
    return df



#Renommer les colonnes X4 et X6
def renommer_les_colonnes(dframe):
    df = dframe.withColumnRenamed('X4', 'match')
    df = df.withColumnRenamed('X6', 'competition')
    return df



#Selectionner les colonnes
def selection_des_colonnes(dframe):
    df = dframe.select('match','competition','adversaire','score_france','score_adversaire','penalty_france','penalty_adversaire','date')
    return df
  
    
#Remplacer les valeurs nulles    
def remplacer_les_valeurs_vulles(dframe):
    df = dframe.withColumn("penalty_france", dframe.penalty_france.cast('int'))
    df = df.withColumn("penalty_adversaire", df.penalty_adversaire.cast('int'))
    return df.na.fill(0)

#Filtrer les match de 1980 à aujourd'hui
def filtrer_match_recent(dframe):
    df = dframe.filter(dframe.date >= '1980-03-01')
    return df

#BOOLEAN Savoir si le match est joué à domicile
def resultat_a_domicile_oui_non(dframe):
    if dframe[0:6] == 'France':
        return True
    else:
        return False


#Rajout de la colonne A Domicile avec la réponse du boolean
def a_domicile(dframe):
    df = dframe.withColumn('a_Domicile', resultat_a_domicile_oui_non(dframe.match));
    return df

resultat_a_domicile_oui_non = F.udf(resultat_a_domicile_oui_non, BooleanType())


#Savoir si ils sont champion du monde
def en_coupe_du_monde(competition_colonne):
    if competition_colonne[:5] == 'Coupe':
        return 1
    else:
        return 0
    
jouer_en_coupe_du_monde = F.udf(en_coupe_du_monde, IntegerType())    
    
def statistiques(dframe):
    df = (dframe
        .groupBy("adversaire")
        .agg(
        F.avg(dframe.score_france).alias("moy_marque_par_france"),
        F.avg(dframe.score_adversaire).alias("moy_marque_adversaire"),
        F.count(dframe.adversaire).alias("match_joue_total"), (F.sum(dframe.a_Domicile.cast('int')) * 100 / F.count(dframe.adversaire)).alias('%_match_a_domicile'),
        F.sum(jouer_en_coupe_du_monde(dframe.competition)).alias('nbr_match_en_coupe_du_monde'),
        F.max(dframe.penalty_france).alias('+_grand_nbr_de_penaltie'),
        (F.sum(dframe.penalty_france) - F.sum(dframe.penalty_adversaire)).alias('penalty_france_-_adversaire')

    )
    )
    return df


if __name__ == "__main__":
    main(sys.argv)