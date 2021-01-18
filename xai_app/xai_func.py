import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report

from catboost import CatBoostClassifier

import pickle

import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
# import eli5
from functools import partial
import numpy as np
import shap

# def create_model(path_from, path_to):
#     df = pd.read_csv(path_from, sep=';')

#     y = df["y"].map({"no": 0, "yes": 1})
#     X = df.drop("y", axis=1)

#     cat_features = [c for c in X.columns if X[c].dtype.name == 'object']
#     num_features = [c for c in X.columns if X[c].dtype.name != 'object']


#     preprocessor = ColumnTransformer([("numerical", "passthrough", num_features),
#                                   ("categorical", OneHotEncoder(sparse=False, handle_unknown="ignore"),
#                                    cat_features)])

#     model = Pipeline([("preprocessor", preprocessor),
#                       # Add a scale_pos_weight to make it balanced
#                       ("model", CatBoostClassifier(n_estimators=1, learning_rate=0.05,scale_pos_weight=(1 - y.mean())))])

#     X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=.3, random_state=42)

#     model.X = X
#     model.y = y
#     model.fit(X_train, y_train)

#     y_pred = model.predict(X_test)

#     # print(classification_report(y_test, y_pred))

#     with open(f'{path_to}', 'wb') as f:
#         pickle.dump(model, f)

#     return list(model.X.columns)

# def open_model(path):
#     with open(path, 'rb') as f:
#         model = pickle.load(f)

#     print(list(model.X.columns))


# def convert_to_lime_format(X, categorical_names, col_names=None, invert=False):

#     # If the data isn't a dataframe, we need to be able to build it
#     if not isinstance(X, pd.DataFrame):
#         X_lime = pd.DataFrame(X, columns=col_names)
#     else:
#         X_lime = X.copy()

#     for k, v in categorical_names.items():
#         if not invert:
#             label_map = {
#                 str_label: int_label for int_label, str_label in enumerate(v)
#             }
#         else:
#             label_map = {
#                 int_label: str_label for int_label, str_label in enumerate(v)
#             }

#         X_lime.iloc[:, k] = X_lime.iloc[:, k].map(label_map)

#     return X_lime


# def custom_predict_proba(X, model, categorical_names):
#     X_str = convert_to_lime_format(X, categorical_names, col_names=model.X.columns, invert=True)
#     return model.predict_proba(X_str)


# def XAI(data, path):
#     with open(path, 'rb') as f:
#         model = pickle.load(f)

#     categorical_names = {}

#     preprocessor = model.named_steps["preprocessor"]
#     cat_features = [c for c in model.X.columns if model.X[c].dtype.name == 'object']
#     ohe_categories = preprocessor.named_transformers_["categorical"].categories_
#     new_ohe_features = [f"{col}__{val}" for col, vals in zip(cat_features, ohe_categories) for val in vals]

#     for col in cat_features:
#         categorical_names[model.X.columns.get_loc(col)] = [new_col.split("__")[1]
#                                                            for new_col in new_ohe_features
#                                                            if new_col.split("__")[0] == col]

#     explainer = LimeTabularExplainer(convert_to_lime_format(model.X, categorical_names).values,
#                                  mode="classification",
#                                  feature_names=model.X.columns.tolist(),
#                                  categorical_names=categorical_names,
#                                  categorical_features=categorical_names.keys(),
#                                  discretize_continuous=True,
#                                  random_state=42)

#     columns = model.X.columns
#     df = pd.DataFrame([data], columns=columns)
#     data = df.iloc[[0], :]

#     observation = convert_to_lime_format(data, categorical_names).values[0]
#     predict_proba = partial(custom_predict_proba, model=model, categorical_names=categorical_names)
#     explanation = explainer.explain_instance(observation, predict_proba, num_features=10)
#     # explanation.show_in_notebook(show_table=True, show_all=False)
#     explanation.save_to_file(path.split('.')[0]+'.html')




def open_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)

    print(list(model.X.columns))



def create_model(path_from, path_to):
    df = pd.read_csv(path_from, sep=',')
    df = df.drop(df.columns[0],axis=1)
    # y = df["y"].map({"no": 0, "yes": 1})
    name_y = list(df.columns)[-1]
    y = df[name_y]
    X = df.drop(name_y, axis=1)

    cat_features = [c for c in X.columns if X[c].dtype.name == 'object']
    num_features = [c for c in X.columns if X[c].dtype.name != 'object']


    model = CatBoostClassifier(cat_features=cat_features,n_estimators=10000, learning_rate=0.05,scale_pos_weight=(1 - y.mean()))

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=.3, random_state=42)

    model.X = X
    model.y = y
    model.fit(X_train, y_train)

    # y_pred = model.predict(X_test)

    # print(classification_report(y_test, y_pred))

    with open(f'{path_to}', 'wb') as f:
        pickle.dump(model, f)

    with open(path_to.split('.')[0]+'_data.pickle', 'wb') as f:
        pickle.dump(df, f)

    return list(model.X.columns)






def cus_predict(X,categorical_names,df,model):

  X_2 = []
  for x in X:
    x = list(x)
    cols = list(df.columns)
    for i,col in enumerate(cols):
        if col in categorical_names:
          zns = categorical_names[col]
          x[i] = zns[int(x[i])]
    X_2.append(x)
  X_2=np.array(X_2)
  print(X_2.shape)
  return model.predict_proba(X_2)






def XAI(data, path):

    print(data)
    dff = pd.read_csv('/home/kirill/Загрузки/train_data (1).csv',sep=',')
    print(dff.columns)
    dff = dff.drop(dff.columns[0],axis=1)
    name_y = list(dff.columns)[-1]
    data = dff.drop(name_y, axis=1).iloc[1].tolist()
    print(data)

    with open(path, 'rb') as f:
        model = pickle.load(f)

    with open(path.split('.')[0]+'_data.pickle', 'rb') as f:
        df = pickle.load(f)


    y = df[name_y]
    X = df.drop(name_y, axis=1)


    cat_features = [c for c in X.columns if X[c].dtype.name == 'object']
    num_features = [c for c in X.columns if X[c].dtype.name != 'object']
    print(cat_features)
    all_features = num_features + cat_features

# lime


    categorical_names = {}
    for col in cat_features:
        categorical_names[col] = list(X[col].unique())

    X_2 = X.copy()
    for col in cat_features:
        temp = dict(zip(categorical_names[col],list(range(len(categorical_names[col])))))
        X_2[col] = X_2[col].map(temp)

    # print(categorical_names)

    categorical_names2 = {}
    columns = list(X.columns)
    for k,v in categorical_names.items():
        index = columns.index(k)
        categorical_names2[index] = categorical_names[k]


    explainer = LimeTabularExplainer(X_2.values,
                                 mode="classification",
                                 feature_names=X_2.columns.tolist(),
                                 categorical_names=categorical_names2,
                                 categorical_features=categorical_names2.keys(),
                                 discretize_continuous=False
                                #  ,
                                #  random_state=42,
                                #  verbose=False
                                #  ,kernel_width=3
                                )

    columns = X.columns

    df = pd.DataFrame([data], columns=columns)
    data = df.iloc[[0], :]
    observation = data.values[0]


    for k,v in categorical_names.items():
        observation[list(columns).index(k)]=v.index(observation[list(columns).index(k)])

    cus_predict2 = partial(cus_predict, categorical_names=categorical_names,df=X,model=model)
    # print(cus_predict2)
    print('observation',observation)
    explanation = explainer.explain_instance(observation,cus_predict2, num_features=5)
    explanation.show_in_notebook(show_table=True, show_all=False)
    explanation.save_to_file(path.split('.')[0]+'_lime.html')
# shap
    shap.initjs()

    explainer = shap.TreeExplainer(model)

    observations = X.sample(300, random_state=42).to_numpy()
    shap_values = explainer.shap_values(observations)

    i = 0

    shap.force_plot(explainer.expected_value, shap_values[i],
                    features=observations[i], feature_names=all_features)

    shap.save_html(path.split('.')[0]+'_shap_1.html', shap.force_plot(explainer.expected_value, shap_values[i],
                    features=observations[i], feature_names=all_features))


    shap.save_html(path.split('.')[0]+'_shap_2.html',shap.force_plot(explainer.expected_value, shap_values,
                features=observations, feature_names=all_features))

    shap.summary_plot(shap_values, features=observations, feature_names=all_features, show=False)
    plt.savefig(path.split('.')[0]+'_shap_3.png', format='png', bbox_inches = 'tight')
    plt.close()


def XAI_shap_detail(path,feature):
    with open(path, 'rb') as f:
        model = pickle.load(f)

    with open(path.split('.')[0]+'_data.pickle', 'rb') as f:
        df = pickle.load(f)

    name_y = list(df.columns)[-1]
    y = df[name_y]
    X = df.drop(name_y, axis=1)


    cat_features = [c for c in X.columns if X[c].dtype.name == 'object']
    num_features = [c for c in X.columns if X[c].dtype.name != 'object']
    all_features = num_features + cat_features


    shap.initjs()

    explainer = shap.TreeExplainer(model)

    observations = X.sample(500, random_state=42).to_numpy()
    shap_values = explainer.shap_values(observations)




    shap_values = explainer.shap_values(observations)
    shap.dependence_plot(feature, shap_values,
                    pd.DataFrame(observations, columns=all_features))
    plt.savefig(path.split('.')[0]+'_shap_4.png', bbox_inches = 'tight')
    plt.close()
