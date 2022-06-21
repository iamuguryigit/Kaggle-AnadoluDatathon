#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def lazy_EDA(df):
    
    # to get a quick analysis of the data
    anadolu_report = sweetviz.analyze(train, target_feat='ARTIS_DURUMU', pairwise_analysis="off")
    anadolu_report.show_html()
    

def data_transform(df):
    
    #df.fillna("99999", inplace=True)
    
    #correction of an input.
    df.loc[df.SOZLESME_KOKENI == "TRANS","SOZLESME_KOKENI"] = "TRANS_C"
    
    # data type conversion.
    df["GELIR"] = df["GELIR"].str.replace(",", ".")
    df["GELIR"] = df["GELIR"].astype(float)
    df["GELIR"] = df["GELIR"].astype(int)
    
    df["COCUK_SAYISI"] = df["COCUK_SAYISI"].astype(int)
    
    df["MUSTERI_SEGMENTI"] = df["MUSTERI_SEGMENTI"].astype(int)
    
    # PARTICIPATION_AGE 
    df["BASLANGIC_YILI"] = df["BASLANGIC_TARIHI"].str[:4].astype(int)
    df["PARTICIPATION_AGE"] = df["BASLANGIC_YILI"] - df["DOGUM_TARIHI"]

    
    df = df.rename(columns={'SUBAT_ODENEN_TU': 'SUBAT_ODENEN_TUTAR'})
    
    months = ["OCAK", "SUBAT", "MART", "NISAN", "MAYIS", "HAZIRAN", "TEMMUZ", "AGUSTOS", "EYLUL", "EKIM", "KASIM", "ARALIK"]

    # check if payment amount higher than preset amount or not.
    for i in months:
        df[i] = (df[i + "_ODENEN_TUTAR"] >= df[i + "_VADE_TUTARI"]).astype(int)

        if i != "OCAK":
            df["CHANGE_" + i] = (df[i + "_VADE_TUTARI"] == df["OCAK_VADE_TUTARI"]).astype(int)
    
    # calculation of irregular payment counts.
    df["TOTAL_COUNT"] = df["OCAK"] + df["SUBAT"] + df["MART"] + df["NISAN"] + df["MAYIS"] + df["HAZIRAN"]                        + df["TEMMUZ"] + df["AGUSTOS"] + df["EYLUL"] + df["EKIM"] + df["KASIM"]                        + df["ARALIK"]
    
    # to check if there is a change in the payment amount and to check when it happened as a new feature together.
    df["CHANGE_COUNT"] = 11 - (df["CHANGE_SUBAT"] + df["CHANGE_MART"] + df["CHANGE_NISAN"] + df["CHANGE_MAYIS"]
                                  + df["CHANGE_HAZIRAN"] + df["CHANGE_TEMMUZ"] + df["CHANGE_AGUSTOS"]
                                  + df["CHANGE_EYLUL"] + df["CHANGE_EKIM"] + df["CHANGE_KASIM"]
                                  + df["CHANGE_ARALIK"])
    
    # if there is a change in payment amount or not.
    df["IS_CHANGED"] = (df["CHANGE_COUNT"] < 11).astype(int)
    
    # to see the change of total payments during the year.
    df["HESAP_DEGER_DEGISIM"] = df["SENE_SONU_HESAP_DEGERI"] - df["SENE_BASI_HESAP_DEGERI"]
    df["HESAP_DEGER_DEGISIM_ORANI"] = (((df["SENE_SONU_HESAP_DEGERI"] / df["SENE_BASI_HESAP_DEGERI"]) - 1) * 100)
    df['HESAP_DEGER_DEGISIM_ORANI'] = df['HESAP_DEGER_DEGISIM_ORANI'].fillna(0)
    df['HESAP_DEGER_DEGISIM_ORANI'].replace([np.inf, -np.inf], 0, inplace=True)
    df['HESAP_DEGER_DEGISIM_ORANI'] = df['HESAP_DEGER_DEGISIM_ORANI'].round(0).astype(int)
    
    # Total payment in a year
    df['TUM_ODENEN'] = df["OCAK_ODENEN_TUTAR"] + df["SUBAT_ODENEN_TUTAR"] + df["MART_ODENEN_TUTAR"] +                       df["NISAN_ODENEN_TUTAR"] + df["MAYIS_ODENEN_TUTAR"] + df["HAZIRAN_ODENEN_TUTAR"] +                       df["TEMMUZ_ODENEN_TUTAR"] + df["AGUSTOS_ODENEN_TUTAR"] + df["EYLUL_ODENEN_TUTAR"] +                       df["EKIM_ODENEN_TUTAR"] + df["KASIM_ODENEN_TUTAR"] + df["ARALIK_ODENEN_TUTAR"]
    
    # Total return 
    df['ANAPARA_GETIRI_ORANI'] = ((df['SENE_SONU_HESAP_DEGERI'] - df['TUM_ODENEN']) /
                                     df['SENE_BASI_HESAP_DEGERI'] - 1) * 100
    df['ANAPARA_GETIRI_ORANI'] = df['ANAPARA_GETIRI_ORANI'].fillna(0)
    df['ANAPARA_GETIRI_ORANI'].replace([np.inf, -np.inf], 0, inplace=True)
    df['ANAPARA_GETIRI_ORANI'] = df['ANAPARA_GETIRI_ORANI'].round(0).astype(int)
    
    # Total return compared to given year's inflation.
    df['ENFLASYON_USTU_GETIRI'] = np.where(df['ANAPARA_GETIRI_ORANI'] > 15, 1, 0)
    
    # Total return compared to the USD&EUR increase of given year.
    df['SEPET_USTU_GETIRI'] = np.where(df['ANAPARA_GETIRI_ORANI'] > 31 , 1, 0)
    
    for i in df.select_dtypes(include="float64").columns:
        df[i] = df[i].astype(int)
        
    # object type transformation for catboost.
    df["OFFICE_ID"] = df["OFFICE_ID"].astype(object)
    df["SIGORTA_TIP"] = df["SIGORTA_TIP"].astype(object)
    df["CINSIYET"] = df["CINSIYET"].astype(object)
    df["MEMLEKET"] = df["MEMLEKET"].astype(object)
    
    
    # Combination of some important features.
    df["K_YK"] = df["KAPSAM_TIPI"].astype(str) + "_" + df["YATIRIM_KARAKTERI"].astype(str)
    df["M_Y"] = df["MUSTERI_SEGMENTI"].astype(str) + "_" + df["YATIRIM_KARAKTERI"].astype(str)
    df["M_GG"] = df["MUSTERI_SEGMENTI"].astype(str) + "_" + df["GELIR"].astype(str)
    df["YK_GG"] = df["YATIRIM_KARAKTERI"].astype(str) + "_" + df["GELIR"].astype(str)
    df["A_YYY"] = df["ARALIK_VADE_TUTARI"].astype(str) + "_" + df["YATIRIM_KARAKTERI"].astype(str)
    df["I_S"] = df["SIGORTA_TIP"].astype(str) + "_" + df["IS_CHANGED"].astype(str)
    
    # Features to be dropped.
    for i in ["SUBAT", "MART", "NISAN", "MAYIS", "HAZIRAN", "TEMMUZ", "AGUSTOS", "EYLUL", "EKIM", "KASIM", "ARALIK"]:
        df.drop("CHANGE_" + i, axis=1, inplace=True)

    # Drop Policy id which is unique in data.
    df = df.drop("POLICY_ID", axis=1)

    df.reset_index(drop = True,inplace = True)
    return df


def train_test_split(df,test_size = 0.25):
    
    X = df.drop("ARTIS_DURUMU", axis=1)
    y = df["ARTIS_DURUMU"] 
    x_train, x_val, y_train, y_val = mod.train_test_split(X, y, test_size=test_size, stratify=y, random_state=0)
    return x_train,x_val,y_train,y_val




def objective(trial, data = data.drop("ARTIS_DURUMU",axis = 1),target = data["ARTIS_DURUMU"]):    
    
    # optuna for hyperparameter optimization.
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.005,0.01,0.015,0.02]),
        'n_estimators': trial.suggest_int('n_estimators', 2000, 10000),
        'bagging_temperature': trial.suggest_int('bagging_temperature', 0, 1),
        'random_strength': trial.suggest_int('random_strength', 0, 1),
        'max_bin': trial.suggest_int('max_bin', 200, 400),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 300),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.0001, 1.0, log = True),
        'subsample': trial.suggest_float('subsample', 0.1, 0.8),
        'random_seed': 42,
        'eval_metric': 'AUC',
        'od_type' : "Iter"
    }
    
    skf = StratifiedKFold(n_splits=5)
    
    cv_score = np.empty(5)
    
    for train_index, test_index in skf.split(data, target):
        
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = target[train_index], target[test_index]
        
        model = CatBoostClassifier(**params)  
        categ_feat_idx = np.where(X_train.dtypes == 'category')[0]
        model.fit(X_train, y_train, cat_features = categ_feat_idx, early_stopping_rounds = 40, verbose = 1000)

        y_pred = model.predict_proba(X_test)[:,1]
        roc_auc = met.roc_auc_score(y_test, y_pred)
        
        cv_score.append(roc_auc)        

    return np.mean(cv_score)



def run_model(x_train,y_train,x_val,y_val,feat_max = 40):

    # running the model.
    model_ctb = CatBoostClassifier(iterations=3000, eval_metric = "AUC",#loss_function='Logloss',
                                   l2_leaf_reg=0.8, od_type='Iter', bagging_temperature = 0.2,random_strength = 0.2,
                                   random_seed=17,max_depth = 5, learning_rate = 0.1,
                                   early_stopping_rounds = 40, class_weights = {0:1,1:3}) #rsm = 1.0)

    categ_feat_idx = np.where(x_train.dtypes == 'object')[0]

    model_ctb.fit(x_train, y_train, cat_features=categ_feat_idx)
    
    train_pred = model_ctb.predict(x_train)
    val_pred = model_ctb.predict(x_val)
    
    print(f'Train-Auc score is {met.roc_auc_score(y_train,train_pred)} and Val-Auc score is {met.roc_auc_score(y_val,val_pred)}')
    
    features = x_val.columns
    importances = model_ctb.feature_importances_
    indices = np.argsort(importances)
    indices = indices[-feat_max:]
    plt.figure(figsize = (20,12))
    plt.title("Catboost Feature Importance")
    plt.barh(range(len(indices)),importances[indices],color = "b",align = "center")
    plt.yticks(range(len(indices)),[features[i] for i in indices])
    
    return train_pred,val_pred

def determine_threshold(train_pred,val_pred,y_train,y_val):
    
    #threshold determination to maximize the F1 score. Test data distribution vs train data distributions are different.
    max_train_f1 = 0
    max_val_f1 = 0
    winner_i_train = 0
    winner_i_val = 0
    
    for i in np.linspace(0.4,0.6,20):
        
        iter_train_result = (train_pred > i).astype(int)
        iter_val_result = (val_pred > i).astype(int)
        
        train_score = met.f1_score(y_train,train_pred)
        val_score = met.f1_score(y_val,val_pred)
        
        if max_train_f1 < train_score :
            max_train_f1 = train_score
            winner_i_train = i
        
        if max_val_f1 < val_score:
            max_val_f1 = val_score
            winner_i_val = i
    
    print(f' For Train: threshold maximizing f1 score is {winner_i_train} /n For Val: threshold maximizing f1 score is {winner_i_val}')
    
    return winner_i_train,winner_i_val



def analyze_results(y_val,val_pred):
    
    # print the model's perfomance.
    
    print(met.classification_report(y_val.values, val_pred))
    
    print(met.confusion_matrix(y_val.values, val_pred))
    
    print(met.roc_curve(y_val.values, val_pred))
    

def get_submission(submission,val_pred,threshold):
    
    # prepare data for the submission.
    
    pred = (val_pred > threshold).astype(int)
    
    submission['ARTIS_DURUMU'] = pred.tolist()
    
    submission.to_csv("submission_anadolu_NAN.csv", index=False)
    
    

#def plotLearningCurve(est, X_train,y_train,X_test,y_test,y_lower = 0.75, y_upper = 1.0):
#     x_values = []
#     y_values_train = [] #train ve testin rmse skorları için 2 tane olması gerek.
#     y_values_test = []
    
#     for i in range(2000,len(X_train),2000): #np.linspace(25,len(X_train),20)  i = int(j)
#         X_train_sample = X_train[:i]  #25 li veri al 25 er arttır. Veri çok büyükse 1000er filan  yapabilirsin.
#         y_train_sample = y_train[:i]
        
#         #compose the model with current train data
#         est.fit(X_train_sample,y_train_sample)
#         # y_values_train.append(mod.cross_val_score(est,X_train,y_train,scoring = "accuracy",cv=5).mean())
        
#         #calculate train score
#         y_pred_train = est.predict(X_train_sample)
#         train_score = met.accuracy_score(y_train_sample,y_pred_train)
#         y_values_train.append(train_score)
        
#         #calculate test score
#         y_pred_test = est.predict(X_test)
#         test_score = met.accuracy_score(y_test,y_pred_test)
#         y_values_test.append(test_score)
        
#         x_values.append(i)  
#     plt.plot(x_values,y_values_train, label = "Train")
#     plt.plot(x_values,y_values_test,label = "Test")
#     plt.xlabel("Train Size")
#     plt.ylabel("Accuracy Score")
#     plt.legend()
#     plt.grid(True)
#     plt.ylim(y_lower,y_upper)
#     plt.show()

    
    
    
    
        
    

