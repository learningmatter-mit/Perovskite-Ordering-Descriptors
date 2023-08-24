#!/usr/bin/python
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import sklearn
from itertools import product
import pickle as pkl
import json
import pandas as pd
import numpy as np
import copy
from sisso_classify import SissoClassifier
from generateDescriptors import generateDescriptors

def Save_Sisso_Experiment(settings,results):
    file_name = ""
    file_name += settings[0] + "_"
    file_name += "DescDim" + str(settings[1]) + "_"
    file_name += "SoMethod" + settings[2] + "_"
    if settings[3]:
        file_name += "Weighted" +"_"
    file_name += ".pkl"
    print("Saving\n")
    print(file_name+"\n")
    out_file = open(file_name, 'wb')
    pkl.dump(results, out_file)
    out_file.close()
    
def Save_Sisso_Experiment_Json(settings,results):
    file_name = "../data/sisso_results/"
    file_name += settings[0] + "_"
    file_name += "DescDim" + str(settings[1]) + "_"
    file_name += "SoMethod" + settings[2] + "_"
    if settings[3]:
        file_name += "Weighted" +"_"
    file_name += ".json"
    print("Saving\n")
    print(file_name+"\n")
    results_dictionary = {}
    results_dictionary["fpr"] = results[0]
    results_dictionary["tpr"] = results[1]
    results_dictionary["AUC"] = results[2]
    
    with open(file_name, 'w') as f:
        json.dump(results_dictionary, f)
    f.close()
    

def extract_input_data(input_data):

    B_ionic_radius_average = []
    B_ionic_radius_diff = []
    B_ionic_radius_multiply = []

    B_ox_state_average = []
    B_ox_state_diff = []
    B_ox_state_multiply = []

    B_electronegativity_average = []
    B_electronegativity_diff = []
    B_electronegativity_multiply = []

    for i in range(len(input_data)):
        curr_row = input_data.iloc[i]
        B_ionic_radius_average.append(curr_row.B_ionic_radius["average"])
        B_ionic_radius_diff.append(curr_row.B_ionic_radius["diff"])
        B_ionic_radius_multiply.append(curr_row.B_ionic_radius["multiply"])

        B_ox_state_average.append(curr_row.B_ox_state["average"])
        B_ox_state_diff.append(curr_row.B_ox_state["diff"])
        B_ox_state_multiply.append(curr_row.B_ox_state["multiply"])

        B_electronegativity_average.append(curr_row.B_electronegativity["average"])
        B_electronegativity_diff.append(curr_row.B_electronegativity["diff"])
        B_electronegativity_multiply.append(curr_row.B_electronegativity["multiply"])

    input_data["B_ionic_radius_average"]=B_ionic_radius_average
    input_data["B_ionic_radius_diff"]=B_ionic_radius_diff
    input_data["B_ionic_radius_multiply"]=B_ionic_radius_multiply

    input_data["B_ox_state_average"]=B_ox_state_average
    input_data["B_ox_state_diff"]=B_ox_state_diff
    input_data["B_ox_state_multiply"]=B_ox_state_multiply

    input_data["B_electronegativity_average"]=B_electronegativity_average
    input_data["B_electronegativity_diff"]=B_electronegativity_diff
    input_data["B_electronegativity_multiply"]=B_electronegativity_multiply

    return input_data

def get_no_dft_data(input_data):

    input_data = extract_input_data(input_data)
    
    input_features = [
                      input_data[["A_ionic_radius","B_ionic_radius_average","B_ionic_radius_diff","B_ionic_radius_multiply"]].to_numpy(dtype=np.float32),
                      input_data[["A_ox_state","B_ox_state_average","B_ox_state_diff","B_ox_state_multiply"]].to_numpy(dtype=np.float32),
                      input_data[["A_electronegativity","B_electronegativity_average","B_electronegativity_diff","B_electronegativity_multiply"]].to_numpy(dtype=np.float32),
                      ]
    
    feature_names = [
                    ['r(A)', 'r(B_ave)', 'r(B_diff)', "r(B)r(B')"],
                    ['z(A)', 'z(B_ave)', 'z(B_diff)', "z(B)z(B')"],
                    ['X(A)', 'X(B_ave)', 'X(B_diff)', "X(B)X(B')"]
                    ]
    
    experimental_labels = np.where(input_data['exp_ordering_type'] == 'rs', 1, 0)
    
    return input_features,feature_names,experimental_labels
    
def get_dft_data(input_data):

    input_data = extract_input_data(input_data)
    
    input_features = [
                      input_data[["A_ionic_radius","B_ionic_radius_average","B_ionic_radius_diff","B_ionic_radius_multiply"]].to_numpy(dtype=np.float32),
                      input_data[["A_ox_state","B_ox_state_average","B_ox_state_diff","B_ox_state_multiply"]].to_numpy(dtype=np.float32),
                      input_data[["A_electronegativity","B_electronegativity_average","B_electronegativity_diff","B_electronegativity_multiply"]].to_numpy(dtype=np.float32),
                      input_data[["dft_rocksalt_prob","dft_normalized_conf_entropy"]].to_numpy(dtype=np.float32)
                      ]
    
    feature_names = [['r(A)', 'r(B_ave)', 'r(B_diff)', "r(B)r(B')"],
                    ['z(A)', 'z(B_ave)', 'z(B_diff)', "z(B)z(B')"],
                    ['X(A)', 'X(B_ave)', 'X(B_diff)', "X(B)X(B')"],
                    ['P_rs', 'E_L_rs', 'Entropy']]
    
    experimental_labels = np.where(input_data['exp_ordering_type'] == 'rs', 1, 0)
    
    return input_features,feature_names,experimental_labels
    
def Run_Sisso_Experiment(input_data_type,dimension,SO_method,is_weighted):
    print("Running Experiment \n")
    #### Cross Validator -- Same As Previous Models
    skf =  StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    #### Load Data 
    
    input_data = pd.read_json("../data/perovskite_ordering_data.json")
    
    if input_data_type == "No_DFT":
        data,names,exp_ordering = get_no_dft_data(input_data)
    elif input_data_type == "DFT":
        data,names,exp_ordering = get_dft_data(input_data)
        
    fpr_test = []
    tpr_test = []
    AUC_test = []
    test_indexes = []
    train_indexes = []
    models = []
    scalers = []
    features = []
    best_1d_do = []
    
    for i, (train_index, test_index) in enumerate(skf.split(exp_ordering,exp_ordering)):
        print("Starting Fold \n")
        
        train_indexes.append(train_index)
        test_indexes.append(test_index)

        [x_des,xVars_des,parents] = generateDescriptors(copy.deepcopy(data),copy.deepcopy(names),ops=2)

        x_train = x_des[train_index]
        x_test = x_des[test_index]

        exp_train = exp_ordering[train_index].reshape(-1)
        exp_test = exp_ordering[test_index].reshape(-1) 

        #### USE SISSO TO SELECT FEATURES
        D = copy.deepcopy(x_train)
        P = copy.deepcopy(exp_train)

        classifier = SissoClassifier(dimension,1000,all_l0_combinations=False,weighted = is_weighted,SO_method = SO_method)
        classifier.fit(D,P)
        
        indices = classifier.l0_selected_indices[-1]
        best_1d_do.append(classifier.sis_selected_indices[0])
        features.append(indices)

        #### EVALUATE FEATURES WITH LOGISTIC REGRESSION

        descriptor_scaler = StandardScaler()
        descriptor_scaler.fit(x_train)
        
        scalers.append(descriptor_scaler)
        
        des_train_norm = descriptor_scaler.transform(x_train)

        model = LogisticRegression()

        sis_selected = des_train_norm[:,indices]
        model.fit(sis_selected.reshape(-1,dimension), exp_train)

        models.append(model)
        
        des_test_norm = descriptor_scaler.transform(x_test)
        sis_selected_test = des_test_norm[:,indices]
        preds_test = model.predict_proba(sis_selected_test.reshape(-1,dimension))

        fpr_test_iter,tpr_test_iter,_ = sklearn.metrics.roc_curve(exp_test, preds_test[:,1])
        curr_test_AUC = sklearn.metrics.auc(fpr_test_iter,tpr_test_iter)

        fpr_test.append(fpr_test_iter.tolist())
        tpr_test.append(tpr_test_iter.tolist())
        AUC_test.append(curr_test_AUC.tolist())
        
    settings = (input_data_type,dimension,SO_method,is_weighted)
    #results = (fpr_test,tpr_test,AUC_test,features,best_1d_do,models,scalers,train_indexes,test_indexes)
    #Save_Sisso_Experiment(settings,results)
    results = (fpr_test,tpr_test,AUC_test)
    Save_Sisso_Experiment_Json(settings,results)
    
    
if __name__ == '__main__':
    ### Possible Settings for Experiements
    data_parameters = ["No_DFT","DFT"]
    descript_dim  = [1,2]
    SO_method_parameters = ["convex_hull","decision_tree"]
    is_weighted_parameters = [True,False]    
    
    experiment_parameters = product(data_parameters,descript_dim,SO_method_parameters,is_weighted_parameters)
    
    for exp_params in experiment_parameters:
        Run_Sisso_Experiment(*exp_params)
    
