import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

import os


class PCAAnalysis:

    def __init__(self, df, target_feature):
        """
        Initialize PCAAnalysis object.
        
        Args:
            feature_names (list): Names of the features, default is None.
        """
        self.df = df
        self.target_feature = target_feature

        self.X = None 
        self.feature_names = None

        self.pca_components = None
        self.pca = None  # Instance of PCA
        self.explained_variance = None

    def select_n_components(self, explained_variance_threshold=0.95):
        """
        Select number of components using PCA and fit the PCA model with selected components.

        Args:
            explained_variance_threshold (float): Threshold for cumulative explained variance, default is 0.95.

        Returns:
            int: Number of components selected.
        """
        
        X = self.df.drop(self.target_feature, axis=1)
        
        X = X.select_dtypes(include='number')
        self.feature_names = X.columns.tolist()
        self.X = X

        pca = PCA().fit(X)
        X_pca = pca.fit_transform(X)

        explained_variance_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(explained_variance_ratio_cumsum) + 1), explained_variance_ratio_cumsum, marker='o', linestyle='--')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Explained Variance by Number of Components')
        plt.show()

        pca_columns = ['PC' + str(c) for c in range(1, X_pca.shape[1]+1)] 
        X_pca = pd.DataFrame(X_pca, index=X.index, columns=pca_columns) 
        explained_variance = pd.Series(dict(zip(X_pca.columns, 100.0*pca.explained_variance_ratio_))).to_frame('explained_variance_pct')

        n_components = np.argmax(explained_variance_ratio_cumsum >= explained_variance_threshold) + 1
        self.pca = PCA(n_components=n_components).fit(X)
        self.pca_components = self.pca.components_
        self.explained_variance = explained_variance # in a clearer dataframe
                
    
    def plot_lowdim(self, hue='yes', dimension='2d'):
        """
        Plot the data in lower dimensions after applying PCA.

        Args:
            hue (str): Whether to use color for different classes, 'yes' or 'no', default is 'yes'.
            dimension (str): Dimension of the plot, either '2d' or '3d', default is '2d'.
        """

        y = self.df[self.target_feature]
        X_pca = self.pca.fit_transform(self.X)
 
        columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
        X_pca_df = pd.DataFrame(X_pca, columns=columns)

        if hue == 'yes':
            unique_labels = np.unique(y)
            dict_colors = {label: plt.cm.tab20(i / len(unique_labels)) for i, label in enumerate(unique_labels)}
            y_colors = [dict_colors[label] for label in y]
        else:
            y_colors = ['grey'] * X_pca.shape[0]  # Tous les points en gris si y est None

        if dimension == '2d':
            plt.figure(figsize=(5, 5))
            plt.scatter(X_pca_df['PC1'], X_pca_df['PC2'], color=y_colors)
            plt.xlabel('PC1 - {:.1f}%'.format(self.explained_variance.loc['PC1', 'explained_variance_pct']))
            plt.ylabel('PC2 - {:.1f}%'.format(self.explained_variance.loc['PC2', 'explained_variance_pct']))
            plt.title('PCA 2D Visualization')
        elif dimension == '3d':
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X_pca_df['PC1'], X_pca_df['PC2'], X_pca_df['PC3'], color=y_colors)
            ax.set_xlabel('PC1 - {:.1f}%'.format(self.explained_variance.loc['PC1', 'explained_variance_pct']))
            ax.set_ylabel('PC2 - {:.1f}%'.format(self.explained_variance.loc['PC2', 'explained_variance_pct']))
            ax.set_zlabel('PC3 - {:.1f}%'.format(self.explained_variance.loc['PC3', 'explained_variance_pct']))
            ax.view_init(elev=15, azim=45)
            plt.title('PCA 3D Visualization')
        else:
            raise ValueError('Dimension must be 2d or 3d')
        
        if not os.path.exists('img'):
            os.makedirs('img')

        plt.savefig('img/pca_visualization.png')
        plt.close()

    def plot_pca_components_contributions(self, n_cols=3):
        """
        Plot bar charts showing contributions of each feature to principal components.

        Args:
            n_cols (int): Number of columns for the subplot grid, default is 3.

        Returns:
            matplotlib.figure.Figure: The figure object.
            list: List of axes objects.
        """
        n_components = self.pca_components.shape[0]

        n_rows = (n_components + n_cols - 1) // n_cols
        fig = plt.figure(figsize=(n_cols * 10, n_rows * 8))
        gs = fig.add_gridspec(n_rows, n_cols)
        axs = []
        for i in range(n_components):
            ax = fig.add_subplot(gs[i])
            ax.bar(self.feature_names, self.pca_components[i], align='center')
            ax.set_title(f'PC{i+1} Contributions')
            ax.set_xticklabels(self.feature_names, rotation=90)
            axs.append(ax)
        fig.tight_layout()
        return fig, axs

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

    def explained_variance_ratio(self):
        """
        Calculate explained variance ratio of PCA.

        Returns:
            numpy.ndarray: Explained variance ratio.
        """
        return self.pca.explained_variance_ratio_