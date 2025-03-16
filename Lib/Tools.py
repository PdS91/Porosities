import random
import pandas as pd
import numpy as np

def conditioned_random_sampling(original_df,n_samples,random_pointer=0):
    
    new_rows = []
    for density in original_df['density'].unique():
        density_rows = original_df[original_df['density'] == density]
        sampled_rows = density_rows.sample(n_samples,replace=True)
        if random_pointer > 0:
            for i in range(random_pointer):
                col_name = f'random_coord_{i}'
                sampled_rows[col_name] = np.random.rand(len(sampled_rows))
                
        new_rows.append(sampled_rows)
    balanced_df = pd.concat(new_rows)
    balanced_df = balanced_df.reset_index(drop=True)
    
    return balanced_df