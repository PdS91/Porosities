import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import re
import os

def extract_microstructures(sample_path,keep_density_doubles=True):
    data_dict = {}
    density_list = []
    for filename in os.listdir(sample_path):
        if filename.endswith(".npy"):
            match = re.search(r"distribution_(\d+)_(\d+\.?\d*)\.npy", filename)
            if match:
                sample_number = int(match.group(1))
                density = float(match.group(2))          
                add = True
                
                if (keep_density_doubles == False) and (density in density_list):
                    
                    add = False
                    
                if add == True:

                    # Load the numpy array
                    array_data = np.load(os.path.join(sample_path, filename))

                    # Store in the dictionary
                    data_dict[sample_number] = PorosityDistribution(array_data, density)
                    density_list.append(density)
                
    return data_dict

def extract_porosities_points(sample_path,keep_density_doubles=True):
    all_data = []  # List to store data from all samples
    density_data = []
    density_list = []
    for filename in os.listdir(sample_path):
        if filename.endswith(".npy"):
            match = re.search(r"distribution_(\d+)_(\d+\.?\d*)\.npy", filename)
            if match:
                sample_number = int(match.group(1))
                density = float(match.group(2))
                
                add = True
                
                if (keep_density_doubles == False) and (density in density_list):
                    
                    add = False
                    
                if add == True:

                    # Load the numpy array
                    array_data = np.load(os.path.join(sample_path, filename))

                    # Reshape and filter data where p == 1
                    df = pd.DataFrame(array_data, columns=['x', 'y', 'z', 'p'])
                    df['density'] = density
                    df['sample_number'] = sample_number
                    filtered_df = df[df['p'] == 1]

                    all_data.append(filtered_df)  # Append filtered data to the list

                    density_data.append([sample_number,density])
                    density_list.append(density)

    # Concatenate all data into a single DataFrame
    final_df = pd.concat(all_data, ignore_index=True)
    density_df = pd.DataFrame(density_data, columns=['sample', 'density']).set_index('sample')
    density_df = density_df[['density']]

    return final_df, density_df


class PorosityDistribution:
    def __init__(self, distribution: np.ndarray,density: float):
        self.distribution = distribution
        self.density = density

    def __str__(self):
        return f"PorosityDistribution(distribution={self.distribution}, density={self.density})"
    
    def as_dataframe(self,porosity=-1):
        df = pd.DataFrame(self.distribution, columns=['x','y','z','porosity'])
        df['density'] = self.density
        if porosity:
            df = df[df['porosity']>=porosity]
        return df

    def as_3Darray(self):
        array = np.reshape(self.distribution,(30,30,30,4))
        vf_channel = np.full((30, 30, 30, 1), self.density)
        return np.concatenate((array, vf_channel), axis=3)

    def as_tensor(self):
        return torch.from_numpy(self.as_3Darray()).float()

    def plot_porosity_distribution(self,porosity=0.97):
        df = self.as_dataframe(porosity=porosity)
        fig = px.scatter_3d(df, x='x', y='y', z='z', color = 'porosity', title=f"Porosity Distribution (Density: {self.density})")
        fig.show()

    def plot_porosity_pairgrid(self):
        df = self.as_dataframe(porosity=0.97)
        g = sns.PairGrid(df.iloc[:,:3], diag_sharey=False)
        g.map_upper(sns.scatterplot, s=5)
        g.map_lower(sns.kdeplot)
        g.map_diag(sns.histplot,bins=30)

    def plot_porosity_histogram(self):
        df = self.as_dataframe(porosity=1)
        fig = px.histogram(df.iloc[:,:3], facet_col='variable', title=f"Porosity Histogram (Density: {self.density})")
        fig.show()

