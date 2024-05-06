import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import seaborn as sns

import umap.umap_ as umap  
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mutual_info_score


import os


class PCAAnalysis:

    def __init__(self, df, target_feature=None):
        """
        Initialize PCAAnalysis object.
        
        Args:
            feature_names (list): Names of the features, default is None.
        """
        self.df = df
        self.target_feature = target_feature

        self.X = None 
        self.feature_names = None

        self.n_components = None
        self.pca_components = None
        self.X_pca = None  # Instance of PCA
        self.pca = None
        self.explained_variance = None

    def select_n_components(self, explained_variance_threshold=0.95):
        """
        Select number of components using PCA and fit the PCA model with selected components.

        Args:
            explained_variance_threshold (float): Threshold for cumulative explained variance, default is 0.95.

        Returns:
            int: Number of components selected.
        """
        
        if self.target_feature:
            X = self.df.drop(self.target_feature, axis=1)
            X = X.select_dtypes(include='number')
        else:
            X = self.df.select_dtypes(include='number')
            
        self.feature_names = X.columns.tolist()
        self.X = X

        pca = PCA().fit(X)

        explained_variance_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(explained_variance_ratio_cumsum) + 1), explained_variance_ratio_cumsum, marker='o', linestyle='--')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Explained Variance by Number of Components')
        plt.show()

        pca_columns = ['PC' + str(c) for c in range(1, X.shape[1]+1)] 
        self.pca_components = pd.DataFrame(pca.components_, columns=pca_columns)
        self.explained_variance = pd.Series(dict(zip(pca_columns, 100.0*pca.explained_variance_ratio_)), name='explained_variance_pct')
        
        n_components = np.argmax(explained_variance_ratio_cumsum >= explained_variance_threshold) + 1
        self.n_components = n_components
        self.pca = PCA(n_components=n_components).fit(X)
        
        X_pca = self.pca.transform(X)
        columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
        X_pca_df = pd.DataFrame(X_pca, columns=columns)
        self.X_pca = X_pca_df

    def plot_lowdim(self, title=None, hue='yes', dimension='2d'):
        """
        Plot the data in lower dimensions after applying PCA.

        Args:
            hue (str): Whether to use color for different classes, 'yes' or 'no', default is 'yes'.
            dimension (str): Dimension of the plot, either '2d' or '3d', default is '2d'.
        """

        if self.target_feature:
            y = self.df[self.target_feature]
        pca = PCA().fit(self.X)
        X_pca = pca.transform(self.X)
    
        columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
        X_pca_df = pd.DataFrame(X_pca, columns=columns)

        if self.target_feature:
            if hue == 'yes':
                unique_labels = np.unique(y)
                dict_colors = {label: plt.cm.tab20(i / len(unique_labels)) for i, label in enumerate(unique_labels)}
                y_colors = [dict_colors[label] for label in y]
            else:
                y_colors = ['grey'] * X_pca.shape[0]  
        else:
                y_colors = ['grey'] * X_pca.shape[0]  # Tous les points en gris si y est None

        if dimension == '2d':
            plt.figure(figsize=(5, 5))
            plt.scatter(X_pca_df['PC1'], X_pca_df['PC2'], color=y_colors)
            plt.xlabel('PC1 - {:.1f}%'.format(pca.explained_variance_ratio_[0]*100))
            plt.ylabel('PC2 - {:.1f}%'.format(pca.explained_variance_ratio_[1]*100))
            plt.title('PCA 2D Visualization')
        elif dimension == '3d':
            if X_pca.shape[1] < 3:
                raise ValueError('Insufficient number of principal components for 3D plot')
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X_pca_df['PC1'], X_pca_df['PC2'], X_pca_df['PC3'], color=y_colors)
            ax.set_xlabel('PC1 - {:.1f}%'.format(pca.explained_variance_ratio_[0]*100))
            ax.set_ylabel('PC2 - {:.1f}%'.format(pca.explained_variance_ratio_[1]*100))
            ax.set_zlabel('PC3 - {:.1f}%'.format(pca.explained_variance_ratio_[2]*100))
            ax.view_init(elev=15, azim=45)
            plt.title('PCA 3D Visualization')
        else:
            raise ValueError('Dimension must be 2d or 3d')
        
        if not os.path.exists('img/pca'):
            os.makedirs('img/pca')

        plt.savefig(f'img/pca/pca_visualization_{title}.png')
        plt.close()

    def plot_pca_components_contributions(self, title=None, n_cols=3):
        """
        Plot bar charts showing contributions of each feature to principal components.

        Args:
            n_cols (int): Number of columns for the subplot grid, default is 3.

        Returns:
            matplotlib.figure.Figure: The figure object.
            list: List of axes objects.
        """
        n_components = self.n_components

        n_rows = (n_components + n_cols - 1) // n_cols
        fig = plt.figure(figsize=(n_cols * 10, n_rows * 8))
        gs = fig.add_gridspec(n_rows, n_cols)
        axs = []
        for i in range(n_components):  # Correction ici: démarrage à 0
            # Correction ici: utilisation correcte de gs pour accéder aux positions
            ax = fig.add_subplot(gs[i // n_cols, i % n_cols]) 
            ax.bar(self.feature_names, self.pca_components['PC' + str(i+1)], align='center')  # Notez l'utilisation de i+1 pour correspondre à vos PC
            ax.set_title(f"PC{i+1} Contributions: {self.explained_variance['PC' + str(i+1)]}")  # Correction ici aussi
            ax.set_xticklabels(self.feature_names, rotation=90)
            axs.append(ax)
        fig.tight_layout()

        if not os.path.exists('img/pca'):
            os.makedirs('img/pca')

        plt.savefig(f'img/pca/pca_components_contributions_{title}.png')
        plt.close()

        return axs

    def plot_correlation_circle(self):
        """
        Plot correlation circle for PCA components.
        """

        pcs = self.pca.components_[:2]
        fig = go.Figure()

        # Add vectors for each feature without showing default names
        for i, feature in enumerate(self.feature_names):
            fig.add_trace(go.Scatter(x=[0, pcs[0, i]], y=[0, pcs[1, i]],
                                    mode='lines',
                                    hoverinfo='text',
                                    text=[None, feature],  # Feature name shown on hover
                                    line=dict(width=2),
                                    showlegend=False))

        # Add reference circle
        theta = np.linspace(0, 2*np.pi, 100)
        x_circle = np.cos(theta)
        y_circle = np.sin(theta)
        fig.add_trace(go.Scatter(x=x_circle, y=y_circle,
                                mode='lines',
                                showlegend=False,
                                line=dict(dash='dash', color='grey')))

        # Plot formatting
        fig.update_layout(title='PCA Correlation Circle',
                        xaxis=dict(scaleanchor="y", scaleratio=1, title='PC1'),
                        yaxis=dict(title='PC2'),
                        width=800, height=800,
                        hovermode='closest')

        fig.show()

    # Methods for assessing PCA
        
    def calculate_reconstruction_error(self):
        """
        Calculate reconstruction error of PCA.

        Args:
            X (numpy.ndarray): Input data.

        Returns:
            float: Reconstruction error.
        """

        X = self.X

        reconstructed_X = np.dot(self.pca.transform(X), self.pca.components_) + self.pca.mean_
        reconstruction_error = np.mean(np.sum((X - reconstructed_X) ** 2, axis=1))
        return reconstruction_error

    


class UMAPAnalysis:
    def __init__(self, df, target_feature=None):
        """
        Initialize UMAPAnalysis object.
        
        Args:
            df (DataFrame): Dataframe containing the dataset.
            target_feature (str): Name of the target feature.
        """
        self.df = df
        self.target_feature = target_feature
        self.X = None
        self.feature_names = None
        self.embedding = None
        self.corr_df = None
    
    def reduce_dimensions(self, n_components=2, metric="euclidean", n_neighbors=10, min_dist=0.1, learning_rate=1, **kwargs):
        """
        Reduce the dimensionality of the data using UMAP.

        Args:
            n_components (int): Number of components for the reduced dimensionality, default is 2.
            **kwargs: Additional arguments to pass to UMAP.

        Returns:
            numpy.ndarray: Embedding of the data in lower-dimensional space.
        """

        if self.target_feature:
            X = self.df.drop(self.target_feature, axis=1)
            X = X.select_dtypes(include='number')
        else:
            X = self.df.select_dtypes(include='number')
        self.feature_names = X.columns.tolist()
        self.X = X

        umap_model = umap.UMAP(n_components=n_components, metric="euclidean", n_neighbors=10, min_dist=0.1, learning_rate=1, **kwargs)
        embedding = umap_model.fit_transform(X)
        self.embedding = embedding

    def plot_lowdim(self, title=None, hue='yes', dimension='2d', continuous='no'):
        """
        Plot the embedding in a lower-dimensional space, either as a 2D or 3D visualization, 
        with options for coloring the points according to a target feature.

        This function supports both categorical and continuous color mappings for the 
        target feature. For 2D visualizations, it generates a scatter plot. For 3D visualizations, 
        it creates an animated rotation of the 3D scatter plot. The output is saved to the 'img' directory.

        Parameters:
        - hue (str): Specifies whether to color the points based on the target feature. 
                    Accepts 'yes' (default) for coloring, 'no' for no coloring.
        - dimension (str): The dimensionality of the plot, either '2d' (default) or '3d'.
        - continuous (str): Determines the type of coloring for the target feature. 
                            Use 'yes' for continuous values and 'no' (default) for categorical.

        Requires:
        - The instance must have an `embedding` attribute, which is a NumPy array of the embedded coordinates.
        - If `hue` is 'yes', the instance should have a `target_feature` attribute specifying the column name 
        in `self.df` to use for coloring. 

        Outputs:
        - For 2D plots, a PNG file named 'umap_2d_visualization.png' is saved in the 'img' directory.
        - For 3D plots, a GIF file named 'umap_3d_animation.gif' showing an animated rotation of the plot 
        is saved in the 'img' directory.

        Note:
        - The function creates the 'img' directory if it does not exist.
        - Color mapping for continuous values uses the 'viridis' colormap.
        - Color mapping for categorical values uses the 'tab20' colormap.
        - The function does not return any value.
        """
        if self.target_feature:
            y = self.df[self.target_feature]
        embedding = self.embedding

        if not os.path.exists('img/umap'):
                os.makedirs('img/umap')

        if dimension == '2d':
            fig, ax = plt.subplots(figsize=(5, 5))
            if self.target_feature and hue == 'yes':
                if continuous == 'yes':
                    norm = plt.Normalize(vmin=min(y), vmax=max(y))
                    cmap = plt.cm.get_cmap('viridis')
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    ax.scatter(embedding[:, 0], embedding[:, 1], c=sm.to_rgba(y))
                    plt.colorbar(sm, ax=ax, label=self.target_feature)
                else:
                    unique_labels = np.unique(y)
                    dict_colors = {label: plt.cm.tab20(i / len(unique_labels)) for i, label in enumerate(unique_labels)}
                    y_colors = [dict_colors[label] for label in y]
                    ax.scatter(embedding[:, 0], embedding[:, 1], c=y_colors)
                    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.tab20(i / len(unique_labels)), markersize=10) for i, label in enumerate(unique_labels)], labels=unique_labels)
            else:
                ax.scatter(embedding[:, 0], embedding[:, 1], color='gray')

            ax.set_xlabel('UMAP Dimension 1')
            ax.set_ylabel('UMAP Dimension 2')
            ax.set_title('UMAP 2D Visualization')

            plt.savefig(f'img/umap/umap_2d_visualization_{title}.png')

            plt.close()
            
        elif dimension == '3d':
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111, projection='3d')
            if self.target_feature and hue == 'yes':
                if continuous == 'yes':
                    norm = plt.Normalize(vmin=min(y), vmax=max(y))
                    cmap = plt.cm.get_cmap('viridis')
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=sm.to_rgba(y))
                    fig.colorbar(sm, ax=ax, label=self.target_feature)
                else:
                    unique_labels = np.unique(y)
                    dict_colors = {label: plt.cm.tab20(i / len(unique_labels)) for i, label in enumerate(unique_labels)}
                    y_colors = [dict_colors[label] for label in y]
                    ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=y_colors)
            else:
                ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], color='gray')

            def update_view(i):
                ax.view_init(elev=10, azim=i)

            ani = FuncAnimation(fig, update_view, frames=range(0, 360, 5), interval=50)
            ani.save(f'img/umap/umap_3d_animation_{title}.gif', writer='pillow')
            plt.close()

    def analyze_feature_correlation(self, threshold_continuous=30):
        """
        Analyze the correlation between original features and UMAP dimensions.
        Includes Pearson correlation coefficients, p-values, Spearman correlation,
        Spearman p-values, and Mutual Information scores for both continuous and discrete features,
        all converted to numeric types and MI scores normalized.

        Returns:
            pandas.DataFrame: DataFrame containing the correlation coefficients, p-values,
                            Spearman correlation, Spearman p-values, and Mutual Information scores 
                            between features and UMAP dimensions, with all values converted to floats and MI normalized.
        """
        umap_coords = self.embedding
        features_df = self.df.select_dtypes(include='number')

        umap_df = pd.DataFrame(umap_coords, columns=[f'UMAP_{i}' for i in range(1, umap_coords.shape[1]+1)])
        combined_df = pd.concat([umap_df, features_df], axis=1)

        data_list = []

        for umap_dim in umap_df.columns:
            for feature in features_df.columns:
                pearson_corr, pearson_p = pearsonr(combined_df[umap_dim], combined_df[feature])
                spearman_corr, spearman_p = spearmanr(combined_df[umap_dim], combined_df[feature])
                
                # Determine if the feature is continuous or discrete
                if len(np.unique(combined_df[feature])) > threshold_continuous: 
                    mutual_info = mutual_info_regression(combined_df[[umap_dim]], combined_df[feature])[0]
                else:
                    mutual_info = mutual_info_score(combined_df[umap_dim].astype(int), combined_df[feature].astype(int))
                
                # Normalization of MI
                n_unique_min = min(len(np.unique(combined_df[umap_dim])), len(np.unique(combined_df[feature])))
                mutual_info_norm = mutual_info / np.log(n_unique_min) if n_unique_min > 1 else 0

                data_list.append({
                    'UMAP Dimension': umap_dim,
                    'Feature': feature,
                    'Pearson_Correlation': pearson_corr,
                    'Pearson_p-value': pearson_p,
                    'Spearman_Correlation': spearman_corr,
                    'Spearman_p-value': spearman_p,
                    'Mutual_Information': mutual_info_norm  
                })

        results_df = pd.DataFrame(data_list)

        for col in ['Pearson_Correlation', 'Pearson_p-value', 'Spearman_Correlation', 'Spearman_p-value', 'Mutual_Information']:
            results_df[col] = pd.to_numeric(results_df[col], errors='coerce')

        self.corr_df = results_df

    def plot_correlation_heatmaps(self, title=None):
        """
        Generate heatmaps for correlation.

        """

        if not os.path.exists('img/umap'):
            os.makedirs('img/umap')

        correlations = [
            ('Pearson_Correlation', 'Heatmap of Pearson Correlation'),
            ('Spearman_Correlation', 'Heatmap of Spearman Correlation'),
            ('Mutual_Information', 'Heatmap of Normalized Mutual Information')
        ]

        fig, axes = plt.subplots(len(correlations), 1, figsize=(10, 8 * len(correlations)))

        for i, (col_name, title) in enumerate(correlations):

            pivot_table = self.corr_df.pivot(index="Feature", columns="UMAP Dimension", values=col_name)
            
            sns.heatmap(pivot_table, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[i])
            axes[i].set_title(title)

        plt.tight_layout()
        plt.savefig(f'img/umap/umap_correlation_heatmaps_{title}.png')
        plt.close()  
