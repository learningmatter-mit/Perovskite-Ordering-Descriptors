import json
import collections
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import LogLocator


plt.rcParams["figure.figsize"] = (13, 8)
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams["xtick.major.size"] = 4
plt.rcParams["ytick.major.size"] = 4
plt.rcParams["ytick.major.width"] = 2
plt.rcParams["xtick.major.width"] = 2
plt.rcParams['text.usetex'] = False
plt.rc('lines', linewidth=3, color='g')
plt.rcParams.update({'font.size': 16})
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['mathtext.fontset'] = 'dejavusans'


B_site_degeneracy = {
    0: 6, # layered
    1: 8, # other
    2: 24, # other
    3: 24, # other
    4: 6, # columnar
    5: 2 # rocksalt
}


def plot_formation_energetics(df, data_type):
    dft_energy_counter = {'rocksalt': [], 'columnar': [], 'layered': [], 'other': []}
    temp_list_dft_highest_normalized_dft_energies = []
    
    if data_type == 'm3gnet':
        m3gnet_energy_counter = {'rocksalt': [], 'columnar': [], 'layered': [], 'other': []}

    for _, row in df.iterrows():
    
        ordering_prototypes = row['ordering_prototype']
        dft_energies = np.array([np.nan if x == '' else x for x in row['dft_energy_per_atom']])
        lowest_dft_energy = np.nanmin(dft_energies)
        highest_dft_energy = np.nanmax(dft_energies)      
        temp_list_dft_highest_normalized_dft_energies.append(highest_dft_energy - lowest_dft_energy)
        temp_dft_energy_counter_other = []

        if data_type == 'm3gnet':
            m3gnet_energies = row['m3gnet_energy_per_atom']
            lowest_m3gnet_energy = np.min(m3gnet_energies)
            temp_m3gnet_energy_counter_other = []

        for k in range(len(dft_energies)):
            if ordering_prototypes[k]['B'][1] == 5:
                dft_energy_counter['rocksalt'].append((dft_energies[k] - lowest_dft_energy))
                if data_type == 'm3gnet':
                    m3gnet_energy_counter['rocksalt'].append((m3gnet_energies[k] - lowest_m3gnet_energy))
            elif ordering_prototypes[k]['B'][1] == 4:
                dft_energy_counter['columnar'].append((dft_energies[k] - lowest_dft_energy))
                if data_type == 'm3gnet':
                    m3gnet_energy_counter['columnar'].append((m3gnet_energies[k] - lowest_m3gnet_energy))
            elif ordering_prototypes[k]['B'][1] == 0:
                dft_energy_counter['layered'].append((dft_energies[k] - lowest_dft_energy))
                if data_type == 'm3gnet':
                    m3gnet_energy_counter['layered'].append((m3gnet_energies[k] - lowest_m3gnet_energy))
            else:
                temp_dft_energy_counter_other.append((dft_energies[k] - lowest_dft_energy))
                if data_type == 'm3gnet':
                    temp_m3gnet_energy_counter_other.append((m3gnet_energies[k] - lowest_m3gnet_energy))

        dft_energy_counter['other'].append(temp_dft_energy_counter_other)        
        if data_type == 'm3gnet':
            m3gnet_energy_counter['other'].append(temp_m3gnet_energy_counter_other)        
        
    sort_index_temp_list_dft_highest_normalized_dft_energies = np.argsort(np.array(temp_list_dft_highest_normalized_dft_energies))
    
    _, axs = plt.subplots(1, 2, figsize=(13, 3.5), gridspec_kw={'width_ratios': [3, 1]})

    
    if data_type == 'dft':
        ylabel = 'DFT energy relative to\nground state (eV/atom)'
        ylim = [-0.01, 0.23]
    elif data_type == 'm3gnet':
        ylabel = 'M3GNet energy relative to\nground state (eV/atom)'
        ylim = [-0.01, 0.15]
    else:
        raise ValueError('data_type must be either dft or m3gnet')

    if data_type == 'dft':
        selected_counter = dft_energy_counter
    else:
        selected_counter = m3gnet_energy_counter

    for k in range(len(sort_index_temp_list_dft_highest_normalized_dft_energies)):
        axs[0].scatter((len(sort_index_temp_list_dft_highest_normalized_dft_energies) - 1 - k), (selected_counter['rocksalt'][sort_index_temp_list_dft_highest_normalized_dft_energies[k]]), marker='o', linewidths=0, s=20, color='red', alpha=0.5)
        axs[0].scatter((len(sort_index_temp_list_dft_highest_normalized_dft_energies) - 1 - k), (selected_counter['columnar'][sort_index_temp_list_dft_highest_normalized_dft_energies[k]]), marker='o', linewidths=0, s=20, color='green', alpha=0.5)
        axs[0].scatter((len(sort_index_temp_list_dft_highest_normalized_dft_energies) - 1 - k), (selected_counter['layered'][sort_index_temp_list_dft_highest_normalized_dft_energies[k]]), marker='o', linewidths=0, s=20, color='blue', alpha=0.5)
        for j in range(3):
            axs[0].scatter((len(sort_index_temp_list_dft_highest_normalized_dft_energies) - 1 - k), (selected_counter['other'][sort_index_temp_list_dft_highest_normalized_dft_energies[k]][j]), marker='o', linewidths=0, s=20, color='grey', alpha=0.5)
    
    axs[0].scatter(-100, -100, color='red', linewidths=0, s=100, alpha=0.5, label='Rocksalt')
    axs[0].scatter(-100, -100, color='green', linewidths=0, s=100, alpha=0.5, label='Columnar')
    axs[0].scatter(-100, -100, color='blue', linewidths=0, s=100, alpha=0.5, label='Layered')
    axs[0].scatter(-100, -100, color='grey', linewidths=0, s=100, alpha=0.5, label='Other')           
    axs[0].set(xlabel='Oxide composition sorted by DFT formation energetics', ylabel=ylabel, xlim=[-1, len(df.index.values)], ylim=ylim);
    axs[0].legend(frameon=False).remove()
    axs[0].set_xticks([])
    
    axs[1].hist(np.array(selected_counter['other']).flatten(), bins=50, range=(0, 0.4), color='grey', alpha=0.5, label='Other', orientation='horizontal');
    axs[1].hist(np.array(selected_counter['layered']).flatten(), bins=50, range=(0, 0.4), color='blue', alpha=0.5, label='Layered', orientation='horizontal');
    axs[1].hist(np.array(selected_counter['columnar']).flatten(), bins=50, range=(0, 0.4), color='green', alpha=0.5, label='Columnar', orientation='horizontal');
    axs[1].hist(np.array(selected_counter['rocksalt']).flatten(), bins=50, range=(0, 0.4), color='red', alpha=0.5, label='Rocksalt', orientation='horizontal');
    axs[1].set(xlabel='Count', ylim=ylim);
    axs[1].legend(frameon=False);
    axs[1].set_yticklabels([])

    plt.tight_layout()
