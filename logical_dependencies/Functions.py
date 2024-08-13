#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")


# In[ ]:


## This function is used to check whether any column in the data has similar values or not
def check_categories(data, features):
    for column in features:
        unique = np.unique(data[column])
        if len(unique) ==1:
            print('For column', column)
            print(len(unique))
    else:
        print('There are no features with same entries')


# In[ ]:


# Function to calculate the Q_scores for a given data
# Input: Data and features for which you want to calculate Q_scores
# Output: List of Q_scores for all the feature pairs 
# If there are $k$ attributes, then $(k^{2} - k)$ $Q$-scores are generated

def Q_metric(data, features):
    # Iterate through all pairs of features
    Q_metric = []
    functional_dependencies = []
    for feature1 in features:
        for feature2 in features:
            if feature1 != feature2:  # Avoid redundant pairs
                # Step 1: Calculate Conditional Probabilities
                conditional_probabilities = data.groupby(feature1)[feature2].value_counts(normalize=True).unstack()
                metric = (conditional_probabilities > 0).sum(axis=1) - 1
                metric = metric.sum().sum()
                if len(conditional_probabilities.columns)==1:# Avoid any feature has same entries
                    print(feature1,feature2)
                    q_metric = 0.0
                else:
                    q_metric = metric / (len(conditional_probabilities.index) * (len(conditional_probabilities.columns) - 1))
                Q_metric.append(q_metric)
                if q_metric==0:
                    fd_columns = conditional_probabilities.index,conditional_probabilities.columns
                    functional_dependencies.append(fd_columns)
                    print(f"Functional dependency: {feature1} -> {feature2}")
    if 0 not in Q_metric:
        print("There are no functional dependencies in the data.")
    if all(num == 1 for num in Q_metric):
        print("There are no logical dependencies in the data.")
    else:
        print("There are some logical dependencies in the data.")
    return Q_metric


# In[ ]:


# Function to plot Q_scores of real and synthetic data
# Input: lists of Q_scores of real and synthetic data 
# Output: Plot of real Q_scores on X-axis and synthetic Q_scores on y-axis, points on the diagonal line represents the common dependencies
# The points on the diagonal line are logically dependent features in both real and synthetic data


def feature_pair_plot(real, synthetic):# Your two lists
    x_values = [round(value, 2) for value in real]  # Replace these values with your own list
    y_values = [round(value, 2) for value in synthetic]  # Replace these values with your own list
    # Plotting the points and drawing a curve
    plt.scatter(x_values, y_values, marker='o',color=custom_palette[2])
    plt.plot([0, 1], [0, 1], linestyle='-',color=custom_palette[0])

    # Set the same limits for both axes
    min_limit = min(min(x_values), min(y_values))
    max_limit = max(max(x_values), max(y_values))
    plt.xlim(min_limit-0.2, max_limit+0.2)
    plt.ylim(min_limit-0.2, max_limit+0.2)
    plt.gca().set_aspect('equal', adjustable='box')
    #ax = plt.gca()  #to remove the box surrounding the plot
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    plt.xlabel('Scores of real data',fontname='DejaVu Sans', fontsize=12)
    plt.ylabel('Scores of synthetic data',fontname='DejaVu Sans', fontsize=12)
    

    # Display the plot
    #plt.savefig('tabddpm.png', dpi=300, bbox_inches='tight')
    plt.show()


# In[ ]:


# Function to Compare logical dependencies of real and synthetic data (only for 0 and 1 probabilities)
# Input : REal and synthetic data along with the list of features
     # Function: Calculates the conditional probabilities for each feature pair in real and synthetic data
     # Extract only the 0 and 1 probabilities and see whether the same probabilities are maintained in the synthetic data at the same position
# Output: returns the percentage of logical dependencies preserved by the saynthetic data with respect to real dataa with df_zero, df_one, real_dependencies,count,match
# df_zero = returns the dataframe with common logical dependencies whose probabilities are 0
# df_one = return the dataframe with common logical dependencies whose probabilities are 1
# real_dependencies = returns the dataframe of all logical dependencies present in real data (both 0 and 1 probabilities)
# count= returns total number of logical dependencies in real data 
# match= returns total number of logical dependencies in synthetic data 


def common_dependencies(real_data,synthetic_data, features):
    # Lists to store combinations with conditional probabilities of 0 and 1
    common_zero_probabilities = []
    common_one_probabilities = []
    ground_truth = []
    count=0
    match=0
    
    

    # Iterate through all pairs of features
    for feature1 in features:
        for feature2 in features:
            if feature1 != feature2:  # Avoid redundant pairs
                # Step 1: Calculate Conditional Probabilities
                conditional_probabilities_real = real_data.groupby(feature1)[feature2].value_counts(normalize=True).unstack()
                conditional_probabilities_synthetic = synthetic_data.groupby(feature1)[feature2].value_counts(normalize=True).unstack()
                                

                # Identify rows with any value exactly equal to 1
                #rows_with_one_real = conditional_probabilities_real.index[conditional_probabilities_real.eq(1).any(axis=1)]
                #rows_with_one_synthetic = conditional_probabilities_synthetic.index[conditional_probabilities_synthetic.eq(1).any(axis=1)]
                

                # Fill NaN values with 0 in rows with any value equal to 1
                conditional_probabilities_real = conditional_probabilities_real.fillna(0)
                conditional_probabilities_synthetic = conditional_probabilities_synthetic.fillna(0)
                
                
                # Step 2: Check for 0 or 1 probabilities and store in lists
                for level1 in conditional_probabilities_real.index:
                    for level2 in conditional_probabilities_real.columns:
                        dummy=None
                        if level1 not in conditional_probabilities_synthetic.index or level2 not in conditional_probabilities_synthetic.columns:
                            dummy=np.float64(0.5)
                            
                        probability_real = conditional_probabilities_real.loc[level1, level2]
                        probability_synthetic = conditional_probabilities_synthetic.loc[level1, level2] if dummy is None else dummy
                        
                        if probability_real==0:
                            count+=1
                            ground_truth.append((feature1, level1, feature2, level2,probability_real))
                            if probability_synthetic==0:
                                common_zero_probabilities.append((feature1, level1, feature2, level2,probability_real,probability_synthetic))
                                match+=1
                            else:
                                pass
                        elif probability_real==1:
                            count+=1
                            ground_truth.append((feature1, level1, feature2, level2,probability_real))
                            if probability_synthetic==1:
                                common_one_probabilities.append((feature1, level1, feature2, level2,probability_real,probability_synthetic))
                                match+=1
                            else:
                                pass
                                
    # Create DataFrames
    df_zero = pd.DataFrame(common_zero_probabilities, columns=['Feature1', 'Level1', 'Feature2', 'Level2', 'Probability_real','Probability_synthetic'])
    df_one = pd.DataFrame(common_one_probabilities, columns=['Feature1', 'Level1', 'Feature2', 'Level2', 'Probability_real','Probability_synthetic'])
    real_dependencies = pd.DataFrame(ground_truth, columns=['Feature1', 'Level1', 'Feature2', 'Level2', 'Probability'])
    
    print('Total number of logical dependencies in real data are:', count)
    print('Total number of logical dependencies in synthetic data data are:', match)
    print('The pecentange of dependencies preserved in synthetic data is :',(match/count)*100)
    return df_zero, df_one, real_dependencies,count,match


# In[ ]:





# In[ ]:


## Function to plot the Q_scores of real and synthetic data generated by seven generative models in a single figure

def feature_pair_plot_all(real, synthetics):
    num_plots = len(synthetics)
    num_rows = 2  # Fixed number of rows for the desired layout
    num_cols = 4  # Fixed number of columns for the desired layout
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6))
    model_names = ['CTGAN', 'CTABGAN', 'CTABGAN Plus', 'TVAE', 'NextConvGeN', 'TabDDPM', 'TabuLa'][:num_plots]

    # Plot real data against each synthetic data list
    for i, synthetic in enumerate(synthetics):
        row = i // num_cols  # Calculate the row index
        col = i % num_cols   # Calculate the column index
        ax = axes[row, col]  # Get the corresponding axis

        x_values = [round(value, 2) for value in real]
        y_values = [round(value, 2) for value in synthetic]
        ax.scatter(x_values, y_values, marker='o', color=custom_palette[2], label=f'Synthetic {i+1}')
        ax.plot([0, 1], [0, 1], linestyle='-', color=custom_palette[0], label='Real')
        min_limit = min(min(x_values), min(y_values))
        max_limit = max(max(x_values), max(y_values))
        ax.set_xlim(min_limit-0.2, max_limit+0.2)
        ax.set_ylim(min_limit-0.2, max_limit+0.2)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('Real', fontname='DejaVu Sans', fontsize=10)
        ax.set_ylabel(f'{model_names[i]} ', fontname='DejaVu Sans', fontsize=10)
        ax.set_xticks(np.arange(-0.2, 1.3, 0.2))  # Set x-axis ticks
        ax.set_yticks(np.arange(-0.2, 1.3, 0.2))  # Set y-axis ticks
        

    # Remove any unused subplots
    for i in range(num_plots, num_rows*num_cols):
        fig.delaxes(axes.flatten()[i])
    
    plt.tight_layout(h_pad=0.5)
    #plt.savefig('Airbnb_Q_metric_plot.png', dpi=700, bbox_inches='tight')    
    plt.show()


# In[ ]:


## Function to plot the histogram of real and synthetic Q_scores and to calculate the KL Divergence between the distributions
# Input: Real and synthetic Q_scores
# Output: Plot with difference in KL divergence


## defining colors
custom_palette = {
    0: (70/255, 114/255, 232/255),   # GoogleBlue
    1: (76/255, 150/255, 77/255),     # GoogleGreen
    2: (238/255, 179/255, 62/255)     # GoogleOrange
}

def plot_scores(real_data, synthetic_data):
    data1 = [round(value, 2) for value in real_data]
    data2 = [round(value, 2) for value in synthetic_data]

    # Creating subplots with multiple histograms
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    bin_edges = np.arange(0.0, 1.1, 0.1)
    
    # Plot histograms
    counts1, bins1, _ = axes[0].hist(data1, bins=bin_edges, color=custom_palette[0], edgecolor='black', alpha=0.7)
    axes[0].set_title('Distribution of real scores', fontname='DejaVu Sans', fontsize=12)

    counts2, bins2, _ = axes[1].hist(data2, bins=bin_edges, color=custom_palette[1], edgecolor='black', alpha=0.7)
    axes[1].set_title('Distribution of synthetic scores', fontname='DejaVu Sans', fontsize=12)
    
    # Convert counts to probabilities
    prob1 = counts1 / len(data1)
    prob2 = (counts2 + 1e-9) / len(data2) 
    
    # Calculate KL divergence
    kl_divergence = entropy(prob1, prob2)
    print("KL Divergence score is:", kl_divergence)

    # Set y-axis limits based on the maximum frequency in both real and synthetic data
    max_freq_real = max(np.histogram(data1, bins=bins1)[0])
    max_freq_synthetic = max(np.histogram(data2, bins=bins2)[0])
    max_freq_combined = max(max_freq_real, max_freq_synthetic)

    for ax in axes:
        ax.set_ylim(0, max_freq_combined + 5)  # Adjust +1 for better visualization

    for ax in axes:
        ax.set_xlabel('Value', fontname='DejaVu Sans', fontsize=12)
        ax.set_ylabel('Frequency', fontname='DejaVu Sans', fontsize=12)
        ax.set_xticks(bin_edges)

    # Adjusting layout for better spacing
    plt.tight_layout()
    
    #plt.savefig('real_vs_tabddpm.png', dpi=300, bbox_inches='tight')

    # Display the figure
    plt.show()

