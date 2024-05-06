import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2, f_regression, mutual_info_regression

from scipy.stats import zscore
from scipy.spatial.distance import mahalanobis
from numpy.linalg import inv

import os


# Avant d'utiliser ces classes, il est essentiel de s'assurer que le dataset est compris
# et que ses variables ont des types bien identifiés


class DfAnalysis:
    """
    A class for performing basic analysis on a pandas DataFrame, because I'm done writing always the same
    code in a loop. That's exhausting.

    Attributes:
        df (pandas.DataFrame): The DataFrame to analyze.
    """

    def __init__(self, df):
        self.df = df
        self.resampled_df = None 

    # Methods for checking and handling data quality issues

    def duplicate_check(self):
        """
        Check for and drop duplicated rows in the DataFrame.
        """
        nb_duplicated_rows = self.df.duplicated().sum()

        if nb_duplicated_rows != 0:
            print(f'There are {nb_duplicated_rows} rows that are duplicated so we need to drop them')
            self.df = self.df.drop_duplicates()
            print(f'After dropping duplicated rows, there are {self.df.shape[0]} rows left')
        else:
            print('No duplicated rows')

    def missing_values_check(self):
        """
        Check for missing values in the DataFrame.
        """
        missing_values_per_row = self.df.isnull().sum(axis=1)
        count_per_missing_value = missing_values_per_row.value_counts().sort_index()

        for missing, rows in count_per_missing_value.items():
            print(f'{rows} row(s) have {missing} missing values')

        total_rows_with_missing_values = (self.df.isnull().any(axis=1)).sum()
        indexes_rows_missing_val = self.df.index[self.df.isnull().any(axis=1)].tolist()
        print(f'Total number of rows with missing values: {total_rows_with_missing_values}')
        print(f'List of indexes of rows with missing values: {indexes_rows_missing_val}')

    def impute_missing_values(self, strategy='mean'):
        """
        Impute missing values in numerical columns of the DataFrame.

        Args:
            strategy (str): The strategy to use for imputation. Must be 'mean', 'median', or 'mode'.
        """
        if strategy == 'mean':
            self.df.select_dtypes(include='number').fillna(self.df.select_dtypes(include='number').mean(), inplace=True)
        elif strategy == 'median':
            self.df.select_dtypes(include='number').fillna(self.df.select_dtypes(include='number').median(), inplace=True)
        elif strategy == 'mode':
            self.df.select_dtypes(include='number').fillna(self.df.select_dtypes(include='number').mode().iloc[0], inplace=True)
        else:
            raise ValueError("Strategy must be 'mean', 'median', or 'mode'")

    # Methods for checking and handling data type issues

    def object_types_check(self):
        """
        Check the data types of object columns in the DataFrame.
        """
        obj_cols = self.df.select_dtypes(include='object').columns
        types = self.df[obj_cols].apply(lambda x: x.apply(type).unique())
        types = pd.DataFrame(types.to_dict()).T
        types.columns = ['Data Type']
        print(types)

    # Methods for statistical analysis

    def skewness_kurtosis(self):
        """
        Calculate the skewness and kurtosis of numerical columns in the DataFrame.
        """
        skew = self.df.select_dtypes(include='number').skew()
        kurt = self.df.select_dtypes(include='number').kurt()
        print(f'Skewness:\n{skew}\n\nKurtosis:\n{kurt}')

    def identify_outliers(self, method='zscore', threshold=3):
        """
        Identify outliers in numerical columns of the DataFrame and return both a list of outlier indices and a DataFrame with the percentage of outliers for each column.

        Args:
            method (str): The method to use for identifying outliers. Options: 'zscore', 'iqr', 'mahalanobis'.
            threshold (int or float): The threshold for identifying outliers, specific to the method used.

        Returns:
            List: Indices of all observations being outliers for at least one feature.
            pandas.DataFrame: A DataFrame containing the percentage of outliers for each numerical column, with the column names as the index.
        """
        numerical_cols = self.df.select_dtypes(include='number')
        outliers = pd.DataFrame(index=self.df.index, columns=numerical_cols.columns).fillna(False)

        if method == 'zscore':
            z_scores = numerical_cols.apply(zscore).abs()
            outliers = z_scores > threshold
        elif method == 'iqr':
            for column in numerical_cols.columns:
                Q1 = numerical_cols[column].quantile(0.25)
                Q3 = numerical_cols[column].quantile(0.75)
                IQR = Q3 - Q1
                outlier_condition = (numerical_cols[column] < (Q1 - 1.5 * IQR)) | (numerical_cols[column] > (Q3 + 1.5 * IQR))
                outliers[column] = outlier_condition
        elif method == 'mahalanobis':
            inv_covmat = inv(numerical_cols.cov())
            mean = numerical_cols.mean().values
            mahalanobis_dist = numerical_cols.apply(lambda row: mahalanobis(row, mean, inv_covmat), axis=1)
            outliers = mahalanobis_dist > threshold
        else:
            raise ValueError("Method must be 'zscore', 'iqr', or 'mahalanobis'")

        # Convert boolean DataFrame to indices of outliers and calculate the percentage of outliers for each column
        outlier_indices = np.flatnonzero(outliers.any(axis=1))
        outlier_percentages = outliers.mean() * 100
        
        return outlier_indices.tolist(), outlier_percentages

    # Methods for visualization

    def plot_histograms(self, title=None, features=[]):
        """
        Save histograms of numerical columns in the DataFrame to the current directory in an img folder.
        """
        if features:
            self.df[features].hist(bins=50, figsize=(20, 15))
        else:
            self.df.select_dtypes(include='number').hist(bins=50, figsize=(20, 15))

        if not os.path.exists('img/hist'):
            os.makedirs('img/hist')

        plt.savefig(f'img/hist/histograms_{title}.png')
        plt.close()

    def correlation_matrix(self, title=None, method=None):
        """
        Save a heatmap of the correlation matrix of numerical columns in the DataFrame to the current directory in an img folder.
        method : {pearson, spearman, kendall, mutual_info}
        """
        numerical_cols = self.df.select_dtypes(include='number').columns

        sns.set_style('white')
        
        if method and method in ['pearson', 'spearman', 'kendall']:
            corr = self.df[numerical_cols].corr(method=method)
        elif method == 'mutual_info':
            mi_matrix = pd.DataFrame(index=numerical_cols, columns=numerical_cols)
            for col in numerical_cols:
                mi = mutual_info_regression(self.df, self.df[col])
                mi_matrix.loc[col, :] = mi
            corr = mi_matrix.astype('float64')
        else:
            corr = self.df[numerical_cols].corr()

        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True
        plt.figure(figsize=(15, 10))
        sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm')

        if not os.path.exists('img/corr'):
            os.makedirs('img/corr')

        plt.savefig(f'img/corr/correlation_matrix_{title}.png')
        plt.close()

    def plot_pairplot(self, title=None, target=None):
        """
        Save pairplot of numerical columns in the DataFrame to the current directory in an img folder.
        Pairplots are made with 'hist' kind that combines computer efficiency and readability.
        
        
        Note: This function needs to be uploaded to take into account hue and other arguments

        Args:
            target (str): The name of the target column if available.
        """
        if target:
            sns.pairplot(self.df, hue=target)
        else:
            numerical_cols = self.df.select_dtypes(include='number').columns
            sns.pairplot(self.df[numerical_cols], kind='hist')
        
        if not os.path.exists('img/pairplots'):
            os.makedirs('img/pairplots')

        plt.savefig(f'img/pairplots/pairplot{"_"+target if target else ""}_{title}.png')
        plt.close()

    def plot_boxplots(self, title=None, n_cols=3):
        """
        Save the boxplots of all numeric columns in the DataFrame in subplots using GridSpec to the current directory in an img folder.

        Args:
            n_cols (int): Number of columns in the subplot grid.

        Returns:
            dict: Dict of axes objects.
        """
        columns = self.df.select_dtypes(include=['number']).columns.tolist()
        
        n_features = len(columns)
        n_rows = (n_features + n_cols - 1) // n_cols
        fig = plt.figure(figsize=(n_cols * 6, n_rows * 4))
        gs = GridSpec(n_rows, n_cols, fig)
        
        axs = {}
        for i, column in enumerate(columns):
            row, col = divmod(i, n_cols)
            ax = fig.add_subplot(gs[row, col])
            sns.boxplot(x=column, data=self.df, ax=ax)
            ax.set_title(f'Boxplot of {column}')
            axs[f'{column}'] = ax
        
        # Create img directory if it does not exist
        if not os.path.exists('img/boxplots'):
            os.makedirs('img/boxplots')

        plt.tight_layout()
        plt.savefig(f'img/boxplots/numeric_boxplots_{title}.png')
        plt.close()
        
        return axs

    def plot_categorical_distribution(self, title=None, n_cols=3):
        """
        Save the distribution of all categorical columns in the DataFrame in subplots using GridSpec to the current directory in an img folder.
        
        Args:
            n_cols (int): Number of columns in the subplot grid.
        
        Returns:
            dict: Dict of axes objects.
        """
        columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        n_features = len(columns)
        n_rows = (n_features + n_cols - 1) // n_cols
        fig = plt.figure(figsize=(n_cols * 6, n_rows * 4))
        gs = GridSpec(n_rows, n_cols, fig)
        
        axs = {}
        for i, column in enumerate(columns):
            row, col = divmod(i, n_cols)
            ax = fig.add_subplot(gs[row, col])
            sns.countplot(x=column, data=self.df, ax=ax)
            ax.set_title(f'Distribution of {column}')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            axs[f'{column}'] = ax
        
        if not os.path.exists('img/cat_dist'):
            os.makedirs('img/cat_dist')

        plt.tight_layout()
        plt.savefig(f'img/cat_dist/categorical_distribution_{title}.png')
        plt.close()
        
        return axs

    # Methods for handling data

    def handle_outliers(self, method='iqr', replacement_method='limit'):
        """
        Handle outliers in numerical columns of the DataFrame by either removing them or replacing them.

        Args:
            method (str): The method to use for identifying outliers. Must be 'iqr' or 'median'.
            replacement_method (str): The method to use for handling outliers - 'limit' replaces outliers with boundary values, 'median' replaces outliers with the median. If None, outliers will be dropped.
        """
        numerical_cols = self.df.select_dtypes(include='number').columns
        
        if method == 'iqr':
            Q1 = self.df[numerical_cols].quantile(0.25)
            Q3 = self.df[numerical_cols].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
        elif method == 'median':
            median = self.df[numerical_cols].median()
            std = self.df[numerical_cols].std()
            lower_bound = median - 3 * std
            upper_bound = median + 3 * std
        else:
            raise ValueError("Method must be 'iqr' or 'median'")
        
        if replacement_method == 'limit':
            for col in numerical_cols:
                self.df[col] = np.where(self.df[col] < lower_bound[col], lower_bound[col], self.df[col])
                self.df[col] = np.where(self.df[col] > upper_bound[col], upper_bound[col], self.df[col])
        elif replacement_method == 'median':
            for col in numerical_cols:
                self.df[col] = np.where((self.df[col] < lower_bound[col]) | (self.df[col] > upper_bound[col]), self.df[col].median(), self.df[col])
        elif replacement_method is None:
            self.df = self.df[~((self.df[numerical_cols] < lower_bound) | (self.df[numerical_cols] > upper_bound)).any(axis=1)]
        else:
            raise ValueError("replacement_method must be 'limit', 'median', or None")
    
    def handle_feature_selection(self, target_column, k = None):
        """
        Perform feature selection using appropriate statistical tests for different types of variables. 
        This method automatically handles both categorical and numerical features and estimates 
        information loss for a given value of k, the number of features to select.

        Important considerations:
        - Non-Linear Relationships: This method primarily uses univariate statistical tests 
        that may not effectively capture non-linear relationships between features and the target variable.
        
        - High-Dimensional Data: In datasets with a large number of features (high dimensionality), 
        these univariate statistical tests can become less reliable due to the curse of dimensionality. 
        The performance may start to degrade when dealing with hundreds of features, especially if the 
        number of observations is not significantly larger than the number of features. 

        - Information Loss: This function estimates the information loss based on the sum of scores 
        of the selected features compared to the total score of all features. A significant loss of 
        information might indicate that the selected subset of features does not adequately represent 
        the original dataset. 

        Parameters:
        - target_column (str): The name of the target column.
        - k (int, optional): The number of features to select. If None, a ValueError is raised. 
        Specify an integer value to select the top k features based on their scores.

        Returns:
        - A tuple containing:
            - List of selected features.
            - DataFrame with scores for each feature, providing insight into their relevance.

        Raises:
        - ValueError: If k is None, indicating that the number of features to select must be specified.
        """

        if k == None:
            raise ValueError("k must not be None")
        
        X = self.df.drop(target_column, axis=1)
        y = self.df[target_column]
        
        # Encode categorical target if it's categorical
        if y.dtype == 'object' or y.dtype.name == 'category':
            y = LabelEncoder().fit_transform(y)

        numerical_cols = X.select_dtypes(include='number').columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Normalize numerical features to ensure chi2 applicability
        if numerical_cols:
            X[numerical_cols] = MinMaxScaler().fit_transform(X[numerical_cols])
        
        feature_scores = pd.DataFrame(columns=['Feature', 'Score'])

        if numerical_cols:
            if y.dtype == 'object' or y.dtype.name == 'category':
                score_func = f_classif
            else:
                score_func = f_regression
            selector_num = SelectKBest(score_func=score_func, k='all')
            selector_num.fit(X[numerical_cols], y)
            scores = selector_num.scores_
            for i, col in enumerate(numerical_cols):
                new_row = pd.DataFrame({'Feature': [col], 'Score': [scores[i]]})
                feature_scores = pd.concat([feature_scores, new_row], ignore_index=True, axis=0)

        # Process categorical features with chi2 or mutual_info_classif/regression
        if categorical_cols:
            X_cat = pd.get_dummies(X[categorical_cols])
            if y.dtype == 'object' or y.dtype.name == 'category':
                score_func = chi2
            else:
                score_func = mutual_info_regression if y.nunique() > 2 else mutual_info_classif
            selector_cat = SelectKBest(score_func=score_func, k='all')
            selector_cat.fit(X_cat, y)
            scores = selector_cat.scores_
            for i, col in enumerate(X_cat.columns):
                new_row = pd.DataFrame({'Feature': [col], 'Score': [scores[i]]})
                feature_scores = pd.concat([feature_scores, new_row], ignore_index=True, axis=0)

        # Sort features by score and select top k
        selected_features = feature_scores.sort_values(by='Score', ascending=False).head(k)['Feature'].tolist()

        # Suggest information loss
        total_score = feature_scores['Score'].sum()
        selected_score = feature_scores.sort_values(by='Score', ascending=False).head(k)['Score'].sum()
        info_loss = 1 - (selected_score / total_score)
        print(f"Suggested information loss for k={k}: {info_loss:.2f}")

        return selected_features, feature_scores


##############################################################################################################


class DataPreprocessing:
    def __init__(self, df, target_feature=None, columns_to_exclude=[], categorical_features=[]):
        """
        Initialize DataPreprocessing object.

        Args:
            df (pandas.DataFrame): Input DataFrame.
            columns_to_exclude (list): Columns to exclude from scaling and normalization.
            categorical_features (list): Categorical features, default is an empty list.
        """
        self.df = df
        self.target_feature = target_feature

        self.columns_to_exclude = columns_to_exclude
        self.categorical_features = categorical_features

        self.scaled_df = None 
        self.normalized_df = None
        self.resampled_df = None 

    def scale_selected_columns(self):
        """
        Scale selected columns using StandardScaler.
        """
        if self.columns_to_exclude or self.categorical_features:
            columns_to_scale = [col for col in self.df.columns if col not in self.columns_to_exclude + self.categorical_features]
        else:
            columns_to_scale = self.df.select_dtypes(include='number').columns.tolist()

        scaler = StandardScaler()
        transformer = ColumnTransformer([('scaler', scaler, columns_to_scale)], remainder='passthrough')

        scaled_array = transformer.fit_transform(self.df)
        # We try here to preserve the order of columns (mind the passthrough)
        columns_after_transform = [col for col in columns_to_scale] + [col for col in self.df.columns if col not in columns_to_scale]
        
        scaled_df = pd.DataFrame(scaled_array, index=self.df.index, columns=columns_after_transform)
        
        # Ensuring types are matching
        for col in columns_to_scale:
            scaled_df[col] = scaled_df[col].astype(self.df[col].dtype)

        self.scaled_df = scaled_df
        
    def normalize_data(self):
        """
        Normalize selected columns using MinMaxScaler.

        Returns:
            pandas.DataFrame: Normalized DataFrame.
        """
        columns_to_normalize = [col for col in self.df.columns if col not in self.columns_to_exclude + self.categorical_features]
        
        normalizer = MinMaxScaler()
        transformer = ColumnTransformer([('normalizer', normalizer, columns_to_normalize)], remainder='passthrough')
        
        normalized_array = transformer.fit_transform(self.df)
        normalized_df = pd.DataFrame(normalized_array, columns=self.df.columns, index=self.df.index)
        
        for col in columns_to_normalize + self.columns_to_exclude + self.categorical_features:
            normalized_df[col] = normalized_df[col].astype(self.df[col].dtype)
        
        self.normalized_df = normalized_df

    def handle_imbalanced_data(self, feature_to_balance=None, method='oversampling'):
        """
        Handle imbalanced data using specified method: oversampling, undersampling, smote, smote_little.

        Args:
            target_column (str): The name of the target column.
            method (str): Method to handle imbalanced data. Options are 'oversampling', 'undersampling', and 'smote'. Default is 'oversampling'.
        """

        X = self.df.drop(feature_to_balance, axis=1)
        y = self.df[feature_to_balance]

        if method == 'oversampling':
            resampler = RandomOverSampler(random_state=42)
        elif method == 'undersampling':
            resampler = RandomUnderSampler(random_state=42)
        elif method == 'smote':
            X = pd.get_dummies(self.df.drop(feature_to_balance, axis=1), drop_first=True)
            resampler = SMOTE(random_state=42)
        elif method == 'smote_little':
            X = pd.get_dummies(self.df.drop(feature_to_balance, axis=1), drop_first=True)
            resampler = SMOTE(random_state=42, k_neighbors=min([y.value_counts().min(), 5])) 
        else:
            raise ValueError("Invalid method. Choose between 'oversampling', 'undersampling', and 'smote'.")

        # Fit and apply the resampling
        X_resampled, y_resampled = resampler.fit_resample(X, y)

        # Create a new DataFrame combining resampled features and target
        resampled_df = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=[feature_to_balance])], axis=1)
        
        self.resampled_df = resampled_df

    def split_data(self, target_column, test_size=0.2, random_state=42):
        """
        Split data into train and test sets.

        Args:
            target_column (str): Target column name.
            test_size (float): Size of the test set, default is 0.2.
            random_state (int): Random seed for reproducibility, default is 42.

        Returns:
            tuple: X_train, X_test, y_train, y_test.
        """
        X = self.scaled_df.drop([target_column] + self.columns_to_exclude, axis=1)
        y = self.scaled_df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        return X_train, X_test, y_train, y_test

    



