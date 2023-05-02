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
    0: 6,
    1: 8,
    2: 24,
    3: 24,
    4: 6,
    5: 2
}


def cross_val_analysis(X, y, n_kfold=5):
    outer_cv = StratifiedKFold(n_splits=n_kfold, random_state=0, shuffle=True)

    result = {
        'actual_classes': np.empty([0], dtype=int),
        'predicted_classes': np.empty([0], dtype=int),
        'roc': {},
        'predict_probs': np.empty([0], dtype=float),
    }
    
    for i, (train_ndx, test_ndx) in enumerate(outer_cv.split(X, y)):
        X_train, y_train, X_test, y_test = X[train_ndx], y[train_ndx], X[test_ndx], y[test_ndx]
        result['actual_classes'] = np.append(result['actual_classes'], y_test)
        model = Pipeline(steps=[('scaler', StandardScaler()), ('regressor', LogisticRegression(max_iter=1000))])        
        model.fit(X_train, y_train)
        prediction = model.predict_proba(X_test)
        result['predict_probs'] = np.append(result['predict_probs'], model.predict_proba(X_test))        
        result['predicted_classes'] = np.append(result['predicted_classes'], model.predict(X_test))
        fpr, tpr, _ = roc_curve(y_test, prediction[:,1])
        result['roc'][i] = {'fpr': fpr, 'tpr': tpr, 'auc': auc(fpr, tpr)}
            
    return result


def roc_auc(result, ax=None, label=None, color=None, xy=None, annotate=False):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for i in range(len(result['roc'].keys())):
        interp_tpr = np.interp(mean_fpr, result['roc'][i]['fpr'], result['roc'][i]['tpr'])
        tprs.append(interp_tpr)
        aucs.append(result['roc'][i]['auc'])
    mean_tpr = np.mean(np.array(tprs), axis=0)
    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)        
    if ax != None:
        if annotate:
            ax.plot(mean_fpr, mean_tpr, color=color, linewidth=4)
            ax.annotate(label + '\n({:.2f})'.format(mean_auc), xy=xy, fontsize=14, xycoords='axes fraction', c=color)
        else:
            ax.plot(mean_fpr, mean_tpr, color=color, label=label + '\n({:.2f})'.format(mean_auc), linewidth=4)
    return mean_auc, std_auc


def plot_boundaries(X, y, ax, descriptor_set, colorbar, exp_ordering_param):
    descriptor_names = {
        'B_ionic_radius_diff': '$\Delta r_{\mathrm{ion(B)}}$',
        'B_ox_state_diff': '$\Delta n_{\mathrm{ox(B)}}$',
        'dft_normalized_conf_entropy': '$S_{\mathrm{conf}}$',
        'dft_rocksalt_prob': '$P_{\mathrm{r}}$'
    }   

    X_train, X_test, y_train, y_test, exp_ordering_param_train, exp_ordering_param_test = train_test_split(X, y, exp_ordering_param, test_size=0.2, random_state=0)
    model = Pipeline(steps=[('scaler', StandardScaler()), ('regressor', LogisticRegression(max_iter=1000))])        
    model.fit(X_train, y_train)
    scaler_means = model[0].mean_
    scaler_stds = np.sqrt(model[0].var_)
    regressor_coefs = model[1].coef_[0]
    regressor_intercept = model[1].intercept_[0]

    if len(descriptor_set) == 2:            
        xlabel = descriptor_names[descriptor_set[0]]
        ylabel = descriptor_names[descriptor_set[1]]
        if ylabel == '$\Delta n_{\mathrm{ox(B)}}$':
            ylabel = '\n$\Delta n_{\mathrm{ox(B)}}$'

        train_plot_x = X_train[:,0]
        train_plot_y = X_train[:,1]
        test_plot_x = X_test[:,0]
        test_plot_y = X_test[:,1]       
 
    elif len(descriptor_set) > 2:        
        scaled_regressor_coefs = regressor_coefs/scaler_stds
        scaled_regressor_intercept = regressor_intercept - np.sum((regressor_coefs*scaler_means)/scaler_stds)               
        dft_indexes = [0, 1]
        atomic_indexes = [2, 3]
        vec_dft = scaled_regressor_coefs[dft_indexes]
        vec_dft_norm = vec_dft.sum()
        vec_dft = vec_dft/vec_dft_norm
        vec_atomic = scaled_regressor_coefs[atomic_indexes]
        vec_atomic_norm = vec_atomic.sum()
        vec_atomic = vec_atomic/vec_atomic_norm
        xlabel = '{:.2f} {} + {:.2f} {}'.format(vec_dft[0], descriptor_names['dft_normalized_conf_entropy'], vec_dft[1], descriptor_names['dft_rocksalt_prob'])
        ylabel = '{:.2f} {} + {:.2f} {}'.format(vec_atomic[0], descriptor_names['B_ionic_radius_diff'], vec_atomic[1], descriptor_names['B_ox_state_diff'])
        
        train_plot_x = np.matmul(X_train[:,dft_indexes], vec_dft)
        train_plot_y = np.matmul(X_train[:,atomic_indexes], vec_atomic)
        test_plot_x = np.matmul(X_test[:,dft_indexes], vec_dft)
        test_plot_y = np.matmul(X_test[:,atomic_indexes], vec_atomic)

    plot_x_all = np.array(list(train_plot_x.flatten()) + list(test_plot_x.flatten()))
    plot_y_all = np.array(list(train_plot_y.flatten()) + list(test_plot_y.flatten()))
    xlim_grid = [-0.2*plot_x_all.max() + 1.2*plot_x_all.min(), 1.2*plot_x_all.max() - 0.2*plot_x_all.min()]
    ylim_grid = [-0.2*plot_y_all.max() + 1.2*plot_y_all.min(), 1.2*plot_y_all.max() - 0.2*plot_y_all.min()]
    xgrid, ygrid = np.meshgrid(np.linspace(xlim_grid[0], xlim_grid[1], 100), np.linspace(ylim_grid[0], ylim_grid[1], 100))
    
    if len(descriptor_set) == 2:            
        zgrid = model[1].predict_proba(model[0].transform(np.hstack((xgrid.reshape(-1,1), ygrid.reshape(-1,1)))))[:,1:2].reshape(100,100)
    elif len(descriptor_set) > 2:        
        zgrid = 1.0/(1.0 + np.exp(-(vec_dft_norm*xgrid + vec_atomic_norm*ygrid + scaled_regressor_intercept)))
    ax.contourf(xgrid, ygrid, zgrid, levels=np.linspace(0,1,1000), cmap=colorbar)

    temp_colors = np.where((exp_ordering_param_train > 0) & (exp_ordering_param_train < 0.5), '#f9d1c1', np.where(y_train==1, '#ef8a62', '#67a9cf'))
    temp_linewidths = np.where((exp_ordering_param_train > 0) & (exp_ordering_param_train < 0.5), 1, 2)
    ax.scatter(train_plot_x, train_plot_y, marker='o', s=100, c=temp_colors, linewidths=temp_linewidths, edgecolors='black', alpha=0.6)
    temp_colors = np.where((exp_ordering_param_test > 0) & (exp_ordering_param_test < 0.5), '#f9d1c1', np.where(y_test==1, '#ef8a62', '#67a9cf'))
    temp_linewidths = np.where((exp_ordering_param_test > 0) & (exp_ordering_param_test < 0.5), 1, 2)
    ax.scatter(test_plot_x, test_plot_y, marker='^', s=100, c=temp_colors, linewidths=temp_linewidths, edgecolors='black', alpha=0.6)
     
    xlim = [-0.1*plot_x_all.max() + 1.1*plot_x_all.min(), 1.1*plot_x_all.max() - 0.1*plot_x_all.min()]
    ylim = [-0.1*plot_y_all.max() + 1.1*plot_y_all.min(), 1.1*plot_y_all.max() - 0.1*plot_y_all.min()]
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim);

    
def plot_boundaries_additional(X, y, ax, descriptor_set, colorbar, exp_ordering_param):
    descriptor_names = {
        'B_ionic_radius_diff': '$\Delta r_{\mathrm{ion(B)}}$',
        'B_ox_state_diff': '$\Delta n_{\mathrm{ox(B)}}$',
        'dft_rocksalt_layered_diff': '$\Delta E_{\mathrm{l,r}}$'
    }   

    X_train, X_test, y_train, y_test, exp_ordering_param_train, exp_ordering_param_test = train_test_split(X, y, exp_ordering_param, test_size=0.2, random_state=0)
    model = Pipeline(steps=[('scaler', StandardScaler()), ('regressor', LogisticRegression(max_iter=1000))])        
    model.fit(X_train, y_train)
    scaler_means = model[0].mean_
    scaler_stds = np.sqrt(model[0].var_)
    regressor_coefs = model[1].coef_[0]
    regressor_intercept = model[1].intercept_[0]
      
    scaled_regressor_coefs = regressor_coefs/scaler_stds
    scaled_regressor_intercept = regressor_intercept - np.sum((regressor_coefs*scaler_means)/scaler_stds)               
    dft_indexes = [0]
    atomic_indexes = [1, 2]
    vec_dft = scaled_regressor_coefs[dft_indexes]
    vec_dft_norm = vec_dft.sum()
    vec_dft = vec_dft/vec_dft_norm
    vec_atomic = scaled_regressor_coefs[atomic_indexes]
    vec_atomic_norm = vec_atomic.sum()
    vec_atomic = vec_atomic/vec_atomic_norm
    xlabel = descriptor_names['dft_rocksalt_layered_diff']
    ylabel = '{:.2f} {} + {:.2f} {}'.format(vec_atomic[0], descriptor_names['B_ionic_radius_diff'], vec_atomic[1], descriptor_names['B_ox_state_diff'])
    
    train_plot_x = np.matmul(X_train[:,dft_indexes], vec_dft)
    train_plot_y = np.matmul(X_train[:,atomic_indexes], vec_atomic)
    test_plot_x = np.matmul(X_test[:,dft_indexes], vec_dft)
    test_plot_y = np.matmul(X_test[:,atomic_indexes], vec_atomic)

    plot_x_all = np.array(list(train_plot_x.flatten()) + list(test_plot_x.flatten()))
    plot_y_all = np.array(list(train_plot_y.flatten()) + list(test_plot_y.flatten()))
    xlim_grid = [-0.2*plot_x_all.max() + 1.2*plot_x_all.min(), 1.2*plot_x_all.max() - 0.2*plot_x_all.min()]
    ylim_grid = [-0.2*plot_y_all.max() + 1.2*plot_y_all.min(), 1.2*plot_y_all.max() - 0.2*plot_y_all.min()]
    xgrid, ygrid = np.meshgrid(np.linspace(xlim_grid[0], xlim_grid[1], 100), np.linspace(ylim_grid[0], ylim_grid[1], 100))
    
    zgrid = 1.0/(1.0 + np.exp(-(vec_dft_norm*xgrid + vec_atomic_norm*ygrid + scaled_regressor_intercept)))
    ax.contourf(xgrid, ygrid, zgrid, levels=np.linspace(0,1,1000), cmap=colorbar)

    temp_colors = np.where((exp_ordering_param_train > 0) & (exp_ordering_param_train < 0.5), '#f9d1c1', np.where(y_train==1, '#ef8a62', '#67a9cf'))
    temp_linewidths = np.where((exp_ordering_param_train > 0) & (exp_ordering_param_train < 0.5), 1, 2)
    ax.scatter(train_plot_x, train_plot_y, marker='o', s=100, c=temp_colors, linewidths=temp_linewidths, edgecolors='black', alpha=0.6)
    temp_colors = np.where((exp_ordering_param_test > 0) & (exp_ordering_param_test < 0.5), '#f9d1c1', np.where(y_test==1, '#ef8a62', '#67a9cf'))
    temp_linewidths = np.where((exp_ordering_param_test > 0) & (exp_ordering_param_test < 0.5), 1, 2)
    ax.scatter(test_plot_x, test_plot_y, marker='^', s=100, c=temp_colors, linewidths=temp_linewidths, edgecolors='black', alpha=0.6)
    
    xlim = [-0.1*plot_x_all.max() + 1.1*plot_x_all.min(), 1.1*plot_x_all.max() - 0.1*plot_x_all.min()]
    ylim = [-0.1*plot_y_all.max() + 1.1*plot_y_all.min(), 1.1*plot_y_all.max() - 0.1*plot_y_all.min()]
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim);


def roc_auc_posthoc(result, ax=None, color=None, label=None):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for i in range(len(result['roc'].keys())):
        interp_tpr = np.interp(mean_fpr, result['roc'][i]['fpr'], result['roc'][i]['tpr'])
        tprs.append(interp_tpr)
        aucs.append(result['roc'][i]['auc'])
    mean_tpr = np.mean(np.array(tprs), axis=0)
    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)        
    if ax != None:
        ax.plot(mean_fpr, mean_tpr, color=color, label=label, linewidth=4)
    return mean_auc, std_auc


def plot_posthoc_analysis(df, axs, i, key, j, threshold, descriptors, plot_settings):
    temp_auc_results = {}
    
    for k, descriptor in enumerate(descriptors.keys()):     
        X = df[descriptors[descriptor]['details']].to_numpy()
        y = df['y_true'].to_numpy()
        cross_val_analysis_result = cross_val_analysis(X, y);
        
        if j == 3:
            label = 'All'
        else:
            label = '<{:.1f}'.format(threshold)            
        
        num_data = X.shape[0]
        mean_auc, std_auc = roc_auc_posthoc(cross_val_analysis_result, axs[k][i], descriptors[descriptor]['colors'][j], label)
        temp_auc_results[descriptor] = {'mean_auc': mean_auc, 'std_auc': std_auc, 'num_data': num_data}
        
        if threshold == 99999:
            axs[k][i].plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.3)
        
        axs[k][i].set(ylabel='True positive rate', xlim=[-0.02, 1.02], ylim=[-0.02, 1.02]);
        axs[k][i].xaxis.set_ticks(np.arange(0, 1.05, 0.2))
        axs[k][i].yaxis.set_ticks(np.arange(0, 1.05, 0.2))
        axs[k][i].sharex=(axs[1][i])
        axs[k][i].sharey=(axs[1][i])
        axs[k][i].legend(frameon=False, title=plot_settings[key]['plot_label']);
        axs[k][i].set(xlabel='False positive rate');
        axs[k][i].set_title(descriptor, size=16)
    
    return temp_auc_results


def plot_boundaries_sisso(X, y, ax, descriptor_set, colorbar, exp_ordering_param, descriptor_names):
    X_train, X_test, y_train, y_test, exp_ordering_param_train, exp_ordering_param_test = train_test_split(X, y, exp_ordering_param, test_size=0.2, random_state=0)
    model = Pipeline(steps=[('scaler', StandardScaler()), ('regressor', LogisticRegression(max_iter=1000))])        
    model.fit(X_train, y_train)
    scaler_means = model[0].mean_
    scaler_stds = np.sqrt(model[0].var_)
    regressor_coefs = model[1].coef_[0]
    regressor_intercept = model[1].intercept_[0]

    xlabel = descriptor_names[descriptor_set[0]]
    ylabel = descriptor_names[descriptor_set[1]]
    train_plot_x = X_train[:,0]
    train_plot_y = X_train[:,1]
    test_plot_x = X_test[:,0]
    test_plot_y = X_test[:,1]       

    plot_x_all = np.array(list(train_plot_x.flatten()) + list(test_plot_x.flatten()))
    plot_y_all = np.array(list(train_plot_y.flatten()) + list(test_plot_y.flatten()))
    xlim_grid = [-0.2*plot_x_all.max() + 1.2*plot_x_all.min(), 1.2*plot_x_all.max() - 0.2*plot_x_all.min()]
    ylim_grid = [-0.2*plot_y_all.max() + 1.2*plot_y_all.min(), 1.2*plot_y_all.max() - 0.2*plot_y_all.min()]
    xgrid, ygrid = np.meshgrid(np.linspace(xlim_grid[0], xlim_grid[1], 100), np.linspace(ylim_grid[0], ylim_grid[1], 100))
    
    zgrid = model[1].predict_proba(model[0].transform(np.hstack((xgrid.reshape(-1,1), ygrid.reshape(-1,1)))))[:,1:2].reshape(100,100)
    ax.contourf(xgrid, ygrid, zgrid, levels=np.linspace(0,1,1000), cmap=colorbar)

    temp_colors = np.where((exp_ordering_param_train > 0) & (exp_ordering_param_train < 0.5), '#f9d1c1', np.where(y_train==1, '#ef8a62', '#67a9cf'))
    temp_linewidths = np.where((exp_ordering_param_train > 0) & (exp_ordering_param_train < 0.5), 1, 2)
    ax.scatter(train_plot_x, train_plot_y, marker='o', s=100, c=temp_colors, linewidths=temp_linewidths, edgecolors='black', alpha=0.6)
    temp_colors = np.where((exp_ordering_param_test > 0) & (exp_ordering_param_test < 0.5), '#f9d1c1', np.where(y_test==1, '#ef8a62', '#67a9cf'))
    temp_linewidths = np.where((exp_ordering_param_test > 0) & (exp_ordering_param_test < 0.5), 1, 2)
    ax.scatter(test_plot_x, test_plot_y, marker='^', s=100, c=temp_colors, linewidths=temp_linewidths, edgecolors='black', alpha=0.6)
     
    xlim = [-0.1*plot_x_all.max() + 1.1*plot_x_all.min(), 1.1*plot_x_all.max() - 0.1*plot_x_all.min()]
    ylim = [-0.1*plot_y_all.max() + 1.1*plot_y_all.min(), 1.1*plot_y_all.max() - 0.1*plot_y_all.min()]
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim);


def get_prototype_from_random_number(num):
    if (num >= 0) and (num < 6):
        return 0
    elif (num >= 6) and (num < 14):
        return 1
    elif (num >= 14) and (num < 38):
        return 2
    elif (num >= 38) and (num < 62):
        return 3
    elif (num >= 62) and (num < 68):
        return 4
    elif (num >= 68) and (num < 70):
        return 5
    else:
        raise Exception("Number range outside [0, 70)!")

        
def averaging_for_sampling(row, prop, rng_range=None, rng_seed=0, override=None, num_range=None):

    temp_probs_energies = []
    temp_props = []

    if rng_range != None:
        rng = np.random.default_rng(seed=rng_seed)
        random_num = get_prototype_from_random_number(rng.integers(rng_range[0], rng_range[1]))

    for i in range(6):
        for j in range(B_site_degeneracy[i]):
            if rng_range != None:
                if i in num_range:
                    k = random_num
                else:
                    k = i
            elif override != None:
                if i in num_range:
                    k = override
                else:
                    k = i
            else:
                k = i
            temp_probs_energies.append(row['dft_energy'][k])
            temp_props.append(row[prop][k])        
    
    temp_probs = softmax(-np.array(list(temp_probs_energies))/(1300*0.0257/300))
    sampled_data = np.sum(np.multiply(np.array(temp_props), np.array(temp_probs)))
 
    return sampled_data


def sampling_test_without_descriptors(df, prop, sampling_types, repeat_random_times=1):
    properties_sampled = {'ordering_averaged': []}
    costs = {}
    
    for sampling_type in sampling_types:
        if 'random' in sampling_type:
            properties_sampled[sampling_type] = {}
            for i in range(repeat_random_times):
                properties_sampled[sampling_type][i] = []
        else:
            properties_sampled[sampling_type] = []
        costs[sampling_type] = 0
    
    for index, row in df.iterrows():    
        properties_sampled['ordering_averaged'].append(averaging_for_sampling(row, prop))
        
        for i in range(repeat_random_times):
            properties_sampled['random'][i].append(averaging_for_sampling(row, prop, rng_range=[0, 70], rng_seed=i, num_range=[0, 1, 2, 3, 4, 5]))
        costs['random'] += 1
        
        properties_sampled['rocksalt'].append(row[prop][5])     
        costs['rocksalt'] += 1
        
        for i in range(repeat_random_times):
            properties_sampled['rocksalt_random'][i].append(averaging_for_sampling(row, prop, rng_range=[0, 68], rng_seed=i, num_range=[0, 1, 2, 3, 4]))
        costs['rocksalt_random'] += 2
        
        properties_sampled['rocksalt_layered'].append(averaging_for_sampling(row, prop, override=0, num_range=[1, 2, 3, 4]))
        costs['rocksalt_layered'] += 2
    
    for sampling_type in sampling_types:
        costs[sampling_type] = costs[sampling_type]/len(df)    
    
    return properties_sampled, costs


def cross_val_analysis_for_sampling(X, y, n_kfold=5):
    outer_cv = StratifiedKFold(n_splits=n_kfold, random_state=0, shuffle=True)

    result = {
        'predicted_classes': np.empty([0], dtype=int),
        'test_ndxs': np.empty([0], dtype=int),
    }
    
    for i, (train_ndx, test_ndx) in enumerate(outer_cv.split(X, y)):
        X_train, y_train, X_test, y_test = X[train_ndx], y[train_ndx], X[test_ndx], y[test_ndx]
        model = Pipeline(steps=[('scaler', StandardScaler()), ('regressor', LogisticRegression(max_iter=1000))])        
        model.fit(X_train, y_train)
        result['predicted_classes'] = np.append(result['predicted_classes'], model.predict(X_test))
        result['test_ndxs'] = np.append(result['test_ndxs'], np.array(test_ndx))
        
    return result


def sampling_test_with_descriptors(df, prop, sampling_types, descriptor_sets, repeat_random_times=1):
    properties_sampled = {'ordering_averaged': []}
    costs = {}
        
    for index, row in df.iterrows():    
        properties_sampled['ordering_averaged'].append(averaging_for_sampling(row, prop))

    for sampling_type in sampling_types:
        if 'random' in sampling_type:
            temp_properties_sampled = {}
            for i in range(repeat_random_times):
                temp_properties_sampled[i] = []
        else:
            temp_properties_sampled = []
        
        costs[sampling_type] = 0
        
        X = df[descriptor_sets[sampling_type]].to_numpy()
        y = df['y_true'].to_numpy()
        cross_val_analysis_result = cross_val_analysis_for_sampling(X, y)
                
        for predicted_class, test_ndx in zip(list(cross_val_analysis_result['predicted_classes']), list(cross_val_analysis_result['test_ndxs'])):
            row = df.iloc[test_ndx]
            
            if predicted_class == 1: # ordered
                if 'random' in sampling_type:
                    for i in range(repeat_random_times):
                        if 'layered' in sampling_type:
                            temp_properties_sampled[i].append(averaging_for_sampling(row, prop, override=0, num_range=[1, 2, 3, 4]))
                        else:
                            temp_properties_sampled[i].append(row[prop][5])
                else:
                    if 'layered' in sampling_type:
                        temp_properties_sampled.append(averaging_for_sampling(row, prop, override=0, num_range=[1, 2, 3, 4]))
                    else:
                        temp_properties_sampled.append(row[prop][5])
                
                if 'dft_atomic_descriptor_rocksalt_layered' in sampling_type:
                    costs[sampling_type] += 2
                elif 'atomic_descriptor_rocksalt' in sampling_type:
                    costs[sampling_type] += 1
                    
            elif predicted_class == 0: # disordered
                if sampling_type == 'atomic_descriptor_rocksalt_random':
                    for i in range(repeat_random_times):
                        temp_properties_sampled[i].append(averaging_for_sampling(row, prop, rng_range=[0, 68], rng_seed=i, num_range=[0, 1, 2, 3, 4]))
                    costs[sampling_type] += 2                                        
                
                elif sampling_type == 'atomic_descriptor_rocksalt_all':
                    temp_properties_sampled.append(averaging_for_sampling(row, prop))
                    costs[sampling_type] += 6                
        
                elif sampling_type == 'dft_atomic_descriptor_rocksalt_layered_random':
                    for i in range(repeat_random_times):
                        temp_properties_sampled[i].append(averaging_for_sampling(row, prop, rng_range=[6, 68], rng_seed=i, num_range=[1, 2, 3, 4]))
                    costs[sampling_type] += 3                   

                elif sampling_type == 'dft_atomic_descriptor_rocksalt_layered_all':
                    temp_properties_sampled.append(averaging_for_sampling(row, prop))
                    costs[sampling_type] += 6
        
        if 'random' in sampling_type:
            properties_sampled[sampling_type] = {}
            for i in range(repeat_random_times):
                properties_sampled[sampling_type][i] = [x for _, x in sorted(zip(list(cross_val_analysis_result['test_ndxs']), temp_properties_sampled[i]))]
        else:
            properties_sampled[sampling_type] = [x for _, x in sorted(zip(list(cross_val_analysis_result['test_ndxs']), temp_properties_sampled))]

        costs[sampling_type] = costs[sampling_type]/len(df)
    
    return properties_sampled, costs
    

def plot_sampling_strategies(properties_sampled, costs, sampling_types, fig, axs, prop_plot_info):
    for i in (range(len(sampling_types))):
        y_true = properties_sampled['ordering_averaged']
        if 'random' in sampling_types[i]:
            y_pred = properties_sampled[sampling_types[i]][0]
        else:
            y_pred = properties_sampled[sampling_types[i]]           
        axs[i].xaxis.set_ticks(np.arange(-10, 10, 2))
        axs[i].yaxis.set_ticks(np.arange(-10, 10, 2))
        axs[i].plot([-100, 100], [-100, 100], 'k--', linewidth=1)
        axs[i].scatter(y_true, y_pred, s=50, color='None', alpha=0.3, edgecolors='black', linewidths=2)
        axs[i].set(xlim=prop_plot_info['lim'], ylim=prop_plot_info['lim'])
        axs[i].sharex=(axs[i])
        axs[i].sharey=(axs[i])
        if i > 0:
            axs[i].set_yticklabels([])
        else:
            axs[i].set(ylabel='\n\nSampled {}'.format(prop_plot_info['name_y']))
        mae = mean_absolute_error(y_true, y_pred)
        axs[i].annotate('MAE = {:.3f}\nCost = {:.2f}'.format(mae, costs[sampling_types[i]]), xy=(0.05, 0.95),
                        xycoords='axes fraction',
                        horizontalalignment='left', verticalalignment='top');
    
    fig.text(0.54, 0.02, 'DFT ordering-averaged {}'.format(prop_plot_info['name_x']), ha='center')
    

def error_cost_tradeoff(properties_sampled, costs, sampling_types, ax, marker):
    for sampling_type in sampling_types:
        cost = costs[sampling_type]
        
        if 'random' in sampling_type:
            mae = []
            for i in properties_sampled[sampling_type].keys():
                y_true = properties_sampled['ordering_averaged']
                y_pred = properties_sampled[sampling_type][i]
                mae.append(mean_absolute_error(y_true, y_pred))
                
            ax.errorbar(cost, np.average(mae), yerr=np.std(mae), fmt=marker, barsabove=True, markersize=20, markeredgecolor='Black', markeredgewidth=2,  ecolor='Black', markerfacecolor='dimgrey', alpha=1, elinewidth=2, capsize=15, capthick=2)               
        
        else:
            y_true = properties_sampled['ordering_averaged']
            y_pred = properties_sampled[sampling_type]
            mae = mean_absolute_error(y_true, y_pred)
            ax.scatter(cost, mae, s=400, color='dimgrey', marker=marker, edgecolors='Black', alpha=1, linewidths=2)
    
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=np.arange(1, 10)*0.1))
    ax.set(xlim=[0.9, 3.3], ylim=[0.006, 0.8]);