import tkinter as tk
from tkinter import simpledialog, messagebox
import matplotlib.pyplot as plt
from EDA_tools import DfAnalysis


# Voici une fonction créée par mes soins qui combine quelques unes des fonctionnalités de la classe EDA_tools
# et qui permet de mener la plupart des étapes usuelles d'une EDA de manière parfaitement automatisée 
# Evidemment c'est une ébauche non fonctionnelle, qui est cependant supposée le devenir sous peu

class Automatize():

    def __init__(self, df):

        self.df = df

    def auto_eda(self):
        # Afficher les histogrammes pour aider à décider du traitement des valeurs manquantes
        df_analysis = DfAnalysis(self.df)
        df_analysis.plot_histograms()
        plt.show()

        # Demander à l'utilisateur comment traiter les valeurs manquantes
        root = tk.Tk()
        root.withdraw()  # Cacher la fenêtre principale
        missing_values_strategy = simpledialog.askstring("Missing Values", "Choose strategy for handling missing values (mean, median, mode):")

        # Traiter les valeurs manquantes en fonction de la stratégie choisie par l'utilisateur
        df_analysis.impute_missing_values(strategy=missing_values_strategy)

        # Afficher la matrice de corrélation pour aider à décider de la sélection des fonctionnalités
        df_analysis.correlation_matrix()
        plt.show()

        # Demander à l'utilisateur s'il souhaite effectuer une sélection de fonctionnalités
        feature_selection_choice = messagebox.askyesno("Feature Selection", "Perform feature selection?")

        if feature_selection_choice:
        
            target_column = simpledialog.askstring("Input", "Enter the name of the target column for feature selection:")
 
            num_features = simpledialog.askinteger("Feature Selection", "Enter number of features to select:")

            selected_features, features_scores = df_analysis.feature_selection(target_column=target_column, k=num_features)
      
        # Traiter les valeurs aberrantes en demandant à l'utilisateur quelle méthode utiliser
        outlier_method = simpledialog.askstring("Outliers", "Choose method for handling outliers (iqr, median):")
        df_analysis.handle_outliers(method=outlier_method)

        # Afficher les résultats ou les résumés de l'analyse
        # Je réfléchis à
        df_analysis.skewness_kurtosis()

        # Afficher les statistiques récapitulatives après traitement
        print("Summary Statistics after preprocessing:")
        print(df_analysis.df.describe())
        print(selected_features, features_scores)

        # Afficher les diagrammes de répartition pour les variables catégorielles après traitement
        df_analysis.plot_categorical_distribution(column='categorical_column')
        df_analysis.plot_pairplot()
        df_analysis.plot_histograms()
        

    def auto_dcleaning(self):
        """
        Automatically clean data through a series of preprocessing steps.
        """
        print("Starting automatic data cleaning process...")

        # Normalise les données 
        self.normalize_data()
        print("Data normalization completed.")

        # Scale les données
        self.scale_selected_columns()
        print("Data scaling completed.")

        # Traitement des données désiquilibrées 
        root = tk.Tk()
        root.withdraw()  
        target_column = simpledialog.askstring("Input", "Enter the name of the target column:")

        # Demande à l'utilisateur s'il veut équilibrer les classes (on pourra ajouter une méthode qui compare les modeles avec k fold stratifié ou non)
        balance_choice = messagebox.askyesno("Balance Classes", "Do you want to balance the classes?")
        if balance_choice:
            self.handle_imbalanced_data(target_column)
            print("Imbalanced data handling completed.")
        else:
            print("Skipping class balancing.")

        # Comme d'habitude on split les données
        test_size = simpledialog.askfloat("Split Data", "Enter test set size (as a fraction, e.g., 0.2 for 20%):")
        X_train, X_test, y_train, y_test = self.split_data(target_column, test_size=test_size)
        print("Data split into train and test sets.")

        print("Automatic data cleaning process completed.")
        print("Summary statistics after preprocessing:")
        print(self.df.describe())

        root.destroy()  # ATTENTION important de détruire la root window

        return X_train, X_test, y_train, y_test