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
from scipy.spatial.distance import cosine
from sklearn.metrics import r2_score


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


def plot_formation_energetics(df, data_type, ax0, ax1):
    dft_energy_counter = {'rocksalt': [], 'columnar': [], 'layered': [], 'other': []}
    temp_list_dft_highest_normalized_dft_energies = []
    
    if data_type == 'm3gnet':
        m3gnet_energy_counter = {'rocksalt': [], 'columnar': [], 'layered': [], 'other': []}

    for _, row in df.iterrows():
    
        ordering_prototypes = row['ordering_prototype']
        dft_energies = dft_energies = np.array([np.nan if x == None else x for x in row['dft_energy_per_atom']])
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
    
    if data_type == 'dft':
        ylabel = 'DFT energy relative to\nground state (eV/atom)'
        ylim = [-0.01, 0.23]
        bins = 50
    elif data_type == 'm3gnet':
        ylabel = 'M3GNet energy relative to\nground state (eV/atom)'
        ylim = [-0.01, 0.15]
        bins = 75
    else:
        raise ValueError('data_type must be either dft or m3gnet')

    if data_type == 'dft':
        selected_counter = dft_energy_counter
    else:
        selected_counter = m3gnet_energy_counter

    for k in range(len(sort_index_temp_list_dft_highest_normalized_dft_energies)):
        ax0.scatter((len(sort_index_temp_list_dft_highest_normalized_dft_energies) - 1 - k), (selected_counter['rocksalt'][sort_index_temp_list_dft_highest_normalized_dft_energies[k]]), marker='o', linewidths=0, s=20, color='red', alpha=0.5)
        ax0.scatter((len(sort_index_temp_list_dft_highest_normalized_dft_energies) - 1 - k), (selected_counter['columnar'][sort_index_temp_list_dft_highest_normalized_dft_energies[k]]), marker='o', linewidths=0, s=20, color='green', alpha=0.5)
        ax0.scatter((len(sort_index_temp_list_dft_highest_normalized_dft_energies) - 1 - k), (selected_counter['layered'][sort_index_temp_list_dft_highest_normalized_dft_energies[k]]), marker='o', linewidths=0, s=20, color='blue', alpha=0.5)
        for j in range(3):
            ax0.scatter((len(sort_index_temp_list_dft_highest_normalized_dft_energies) - 1 - k), (selected_counter['other'][sort_index_temp_list_dft_highest_normalized_dft_energies[k]][j]), marker='o', linewidths=0, s=20, color='grey', alpha=0.5)
    
    ax0.scatter(-100, -100, color='red', linewidths=0, s=100, alpha=0.5, label='Rocksalt')
    ax0.scatter(-100, -100, color='green', linewidths=0, s=100, alpha=0.5, label='Columnar')
    ax0.scatter(-100, -100, color='blue', linewidths=0, s=100, alpha=0.5, label='Layered')
    ax0.scatter(-100, -100, color='grey', linewidths=0, s=100, alpha=0.5, label='Other')           
    ax0.set(xlabel='Oxide composition sorted by DFT formation energetics', ylabel=ylabel, xlim=[-1, len(df.index.values)], ylim=ylim);
    ax0.legend(frameon=False).remove()
    ax0.set_xticks([])
    
    ax1.hist(np.array(selected_counter['other']).flatten(), bins=bins, range=(0, 0.4), color='grey', alpha=0.5, label='Other', orientation='horizontal');
    ax1.hist(np.array(selected_counter['layered']).flatten(), bins=bins, range=(0, 0.4), color='blue', alpha=0.5, label='Layered', orientation='horizontal');
    ax1.hist(np.array(selected_counter['columnar']).flatten(), bins=bins, range=(0, 0.4), color='green', alpha=0.5, label='Columnar', orientation='horizontal');
    ax1.hist(np.array(selected_counter['rocksalt']).flatten(), bins=bins, range=(0, 0.4), color='red', alpha=0.5, label='Rocksalt', orientation='horizontal');
    ax1.set(xlabel='Count', ylim=ylim);
    ax1.legend(frameon=False, labelcolor='linecolor');
    ax1.set_yticklabels([])


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


def plot_confusion_matrix(df, ax, X, label):
    sorted_labels = [0, 1]
    plot_labels = ['Disordered', 'Ordered']

    cross_val_analysis_result = cross_val_analysis(X, df['y_true'].to_numpy());
    matrix = confusion_matrix(cross_val_analysis_result['actual_classes'], cross_val_analysis_result['predicted_classes'], labels=sorted_labels)
    sns.heatmap(matrix, vmin=0, vmax=max(list(collections.Counter(list(cross_val_analysis_result['actual_classes'])).values())), annot=True, xticklabels=plot_labels, yticklabels=plot_labels, cmap='Greys', fmt='g', cbar=False, ax=ax)
    ax.set(xlabel='Predicted', ylabel='Experimental');
    ax.set_title(label, size=16)


def plot_decision_boundary(df, ax, X, labels):
    colorbar = LinearSegmentedColormap.from_list('colorbar_decision', (
        (0.000, (0.588, 0.765, 0.902)),
        (0.500, (1.000, 1.000, 1.000)),
        (1.000, (0.910, 0.686, 0.565)))
    )

    X_train, X_test, y_train, y_test, exp_ordering_param_train, exp_ordering_param_test = train_test_split(
        X, df['y_true'].to_numpy(), df['exp_ordering_parameter'].to_numpy(),
        test_size=0.2, random_state=0
    )

    model = Pipeline(steps=[('scaler', StandardScaler()), ('regressor', LogisticRegression(max_iter=1000))])        
    model.fit(X_train, y_train)

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
    ax.scatter(train_plot_x, train_plot_y, marker='o', s=200, c=temp_colors, linewidths=temp_linewidths, edgecolors='black', alpha=0.9)
    temp_colors = np.where((exp_ordering_param_test > 0) & (exp_ordering_param_test < 0.5), '#f9d1c1', np.where(y_test==1, '#ef8a62', '#67a9cf'))
    temp_linewidths = np.where((exp_ordering_param_test > 0) & (exp_ordering_param_test < 0.5), 1, 2)
    ax.scatter(test_plot_x, test_plot_y, marker='^', s=200, c=temp_colors, linewidths=temp_linewidths, edgecolors='black', alpha=0.9)
     
    xlim = [-0.1*plot_x_all.max() + 1.1*plot_x_all.min(), 1.1*plot_x_all.max() - 0.1*plot_x_all.min()]
    ylim = [-0.1*plot_y_all.max() + 1.1*plot_y_all.min(), 1.1*plot_y_all.max() - 0.1*plot_y_all.min()]
    ax.set(xlabel=labels[0], ylabel=labels[1], xlim=xlim, ylim=ylim);


def plot_roc_curve(df, ax, Xs, labels, colors):
    for i in range(len(Xs)):
        tprs = []
        aucs = []
        cross_val_analysis_result = cross_val_analysis(Xs[i], df['y_true'].to_numpy());
        mean_fpr = np.linspace(0, 1, 100)
        for j in range(len(cross_val_analysis_result['roc'].keys())):
            interp_tpr = np.interp(mean_fpr, cross_val_analysis_result['roc'][j]['fpr'], cross_val_analysis_result['roc'][j]['tpr'])
            tprs.append(interp_tpr)
            aucs.append(cross_val_analysis_result['roc'][j]['auc'])
        mean_tpr = np.mean(np.array(tprs), axis=0)
        mean_tpr[0] = 0.0
        mean_tpr[-1] = 1.0
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color=colors[i], label=labels[i] + '\n({:.2f})'.format(mean_auc), linewidth=4)
    
    ax.legend(loc='lower right', fontsize=16, frameon=False, labelcolor='linecolor')
    ax.plot([0, 1], [0, 1], '--', color='black', linewidth=1, alpha=0.3)
    ax.set(xlabel='False positive rate', ylabel='True positive rate', xlim=[-0.02, 1.02], ylim=[-0.02, 1.02])


def plot_correlation_matrix(df, data_type, ax):
    df_cut = pd.DataFrame(
        np.concatenate([
            df[[data_type + '_rocksalt_prob']].to_numpy(), 
            df[[data_type + '_normalized_conf_entropy']].to_numpy(),
            df[[data_type + '_rocksalt_layered_diff']].to_numpy(),
            df['B_ionic_radius'].apply(lambda x: x['diff']).to_numpy().reshape(-1, 1), 
            df['B_ox_state'].apply(lambda x: x['diff']).to_numpy().reshape(-1, 1),            
            df['B_electronegativity'].apply(lambda x: x['diff']).to_numpy().reshape(-1, 1)            
        ], axis=1),
        columns=[
            '$P_{\mathrm{r}}$', '$S_{\mathrm{conf}}$', '$\Delta E_{\mathrm{l,r}}$',
            '$\Delta r_{\mathrm{ion(B)}}$', '$\Delta n_{\mathrm{ox(B)}}$', '$\Delta \chi_{\mathrm{(B)}}$'
        ]
    )

    corr = df_cut.corr(method='pearson')
    mask = ~np.tril(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    ticklabels = df_cut.columns.values
    
    heatmap = sns.heatmap(
        corr, mask=mask, cmap=cmap, ax=ax, vmax=1, vmin=-1, center=0, square=True, annot=True, 
        linewidths=.5, cbar_kws={"shrink": .5}, xticklabels=ticklabels, yticklabels=ticklabels
        )
    heatmap.collections[0].colorbar.set_ticks([-1,0,1])


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
    
    for _, row in df.iterrows():    
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


def sampling_test_with_descriptors(df, prop, sampling_types, repeat_random_times=1):
    descriptor_sets = {
        'dft_rocksalt_layered_random': ['dft_rocksalt_layered_diff'], 
        'dft_rocksalt_layered_all': ['dft_rocksalt_layered_diff'], 
        'm3gnet_rocksalt_random': ['m3gnet_rocksalt_prob', 'm3gnet_normalized_conf_entropy'], 
        'm3gnet_rocksalt_all': ['m3gnet_rocksalt_prob', 'm3gnet_normalized_conf_entropy']
    }
    
    properties_sampled = {'ordering_averaged': []}
    costs = {}
        
    for _, row in df.iterrows():    
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
                
                if 'dft_rocksalt_layered' in sampling_type:
                    costs[sampling_type] += 2
                elif 'm3gnet_rocksalt' in sampling_type:
                    costs[sampling_type] += 1
                    
            elif predicted_class == 0: # disordered
                if sampling_type == 'm3gnet_rocksalt_random':
                    for i in range(repeat_random_times):
                        temp_properties_sampled[i].append(averaging_for_sampling(row, prop, rng_range=[0, 68], rng_seed=i, num_range=[0, 1, 2, 3, 4]))
                    costs[sampling_type] += 2                                        
                
                elif sampling_type == 'm3gnet_rocksalt_all':
                    temp_properties_sampled.append(averaging_for_sampling(row, prop))
                    costs[sampling_type] += 6                
        
                elif sampling_type == 'dft_rocksalt_layered_random':
                    for i in range(repeat_random_times):
                        temp_properties_sampled[i].append(averaging_for_sampling(row, prop, rng_range=[6, 68], rng_seed=i, num_range=[1, 2, 3, 4]))
                    costs[sampling_type] += 3                   

                elif sampling_type == 'dft_rocksalt_layered_all':
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


def plot_sampling_results(df, sampling_types, prop, axs, axis_name, axis_lim):
    
    for i in (range(len(sampling_types))):
        sampling_types_cut = sampling_types[i]
        
        if i == 0:
            properties_sampled, costs = sampling_test_without_descriptors(df, prop, sampling_types_cut)
        else:
            properties_sampled, costs = sampling_test_with_descriptors(df, prop, sampling_types_cut)

        for i in (range(len(sampling_types_cut))):
            sampling_type = sampling_types_cut[i]
            y_true = properties_sampled['ordering_averaged']

            if 'random' in sampling_type:
                y_pred = properties_sampled[sampling_type][0]
            else:
                y_pred = properties_sampled[sampling_type]

            axs[sampling_type].xaxis.set_ticks(np.arange(-10, 10, 2))
            axs[sampling_type].yaxis.set_ticks(np.arange(-10, 10, 2))
            axs[sampling_type].plot([-100, 100], [-100, 100], '--', color='black', linewidth=1)
            axs[sampling_type].scatter(y_pred, y_true, s=20, color='None', alpha=0.2, edgecolors='black', linewidths=2)
            axs[sampling_type].set(xlim=axis_lim, ylim=axis_lim, ylabel= 'Ordering-averaged\n{}'.format(axis_name), xlabel='Sampled\n{}'.format(axis_name))
            mae = mean_absolute_error(y_true, y_pred)
            axs[sampling_type].set_title('MAE = {:.3f}\nCost = {:.2f}'.format(mae, costs[sampling_type]), size=16)

