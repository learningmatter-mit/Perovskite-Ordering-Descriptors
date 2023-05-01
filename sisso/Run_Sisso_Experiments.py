#!/usr/bin/python
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import sklearn
from itertools import product
import pickle as pkl
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
    
    
def Run_Sisso_Experiment(input_data,dimension,SO_method,is_weighted):
    print("Running Experiment \n")
    #### Cross Validator -- Same As Previous Models
    skf =  StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    #### Load Data 
    if input_data == "No_DFT":
        with open('data_no_dft.pkl', 'rb') as f:
            data,names,exp_ordering = pkl.load(f)
    elif input_data == "DFT":
        with open('data_dft.pkl', 'rb') as f:
            data,names,exp_ordering = pkl.load(f)
        
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

        exp_train = exp_ordering[train_index].to_numpy().reshape(-1)
        exp_test = exp_ordering[test_index].to_numpy().reshape(-1) 

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

        fpr_test.append(fpr_test_iter)
        tpr_test.append(tpr_test_iter)
        AUC_test.append(curr_test_AUC)
        
    settings = (input_data,dimension,SO_method,is_weighted)
    results = (fpr_test,tpr_test,AUC_test,features,best_1d_do,models,scalers,train_indexes,test_indexes)
    Save_Sisso_Experiment(settings,results)
    
    
    
if __name__ == '__main__':
    ### Possible Settings for Experiements
    data_parameters = ["No_DFT","DFT"]
    descript_dim  = [1,2]
    SO_method_parameters = ["convex_hull","decision_tree"]
    is_weighted_parameters = [True,False]    
    
    experiment_parameters = product(data_parameters,descript_dim,SO_method_parameters,is_weighted_parameters)
    
    for exp_params in experiment_parameters:
        Run_Sisso_Experiment(*exp_params)
    
    