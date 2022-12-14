from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.linear_model import LogisticRegression


def transform_df(X, columns_to_dummify, features=["Pclass"], thres=10):
    X = convert_df_columns( X, features, type_var="object")
    X["is_child"] = X["Age"].apply(lambda x: 0 if x < thres else 1)
    X["title"] = X["Name"].apply(lambda x: x.split(",")[1].split(".")[0].strip())
    X['surname'] = X['Name'].map(lambda x: '(' in x)
    for col in columns_to_dummify:
        X_dummies = pd.get_dummies(X[col], prefix=col,
                                   drop_first=False, dummy_na=False, prefix_sep='_')
        X = X.join(X_dummies).drop(col, axis=1)
    return X.drop("Name", axis=1).drop("Age", axis=1)


def convert_df_columns( data_str, features, type_var) :
    for feature in features:
        data_str[feature] =  data_str[feature].astype(type_var)
    return data_str


def My_model ( X, y, size, RdomState = 42) :
    #X, y
    X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=size, 
                                                       random_state=RdomState )
    model = LogisticRegression(random_state= RdomState)
    model.fit(X_train, y_train)
    # Run the model
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    score_train = model.score(X_train, y_train)
    score_test = model.score(X_test, y_test)
    
    return {"y_test": y_test, "prediction": y_pred, "proba":y_prob,
           "score_train": score_train, "score_test": score_test,
           "model": model}


def parse_model(X, use_columns):
    if "Survived" not in X.columns:
        raise ValueError("target column survived should belong to df")
    target = X["Survived"]
    X = X[use_columns]
    return X, target