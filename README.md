# Ex-07-Feature-Selection

# AIM:

To Perform the various feature selection techniques on a dataset and save the data to a file.

# EXPLANATION:

Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.

# ALGORITHM:

### STEP 1

Read the given Data

### STEP 2

Clean the Data Set using Data Cleaning Process

### STEP 3

Apply Feature selection techniques to all the features of the data set

### STEP 4

Save the data to the file

# PROGRAM:

# CODE FOR “CarPrice.csv”:

    #Import necessary libraries

    import pandas as pd

    import numpy as np

    from sklearn.model_selection import train_test_split

    from sklearn.preprocessing import LabelEncoder

    from sklearn.feature_selection import SelectKBest, f_regression

    from sklearn.ensemble import ExtraTreesRegressor

    #Load the dataset

    df = pd.read_csv('CarPrice.csv')

    #Drop unnecessary columns

    df = df.drop(['car_ID', 'CarName'], axis=1)

    #Convert categorical columns into numerical columns

    le = LabelEncoder()

    df['fueltype'] = le.fit_transform(df['fueltype'])

    df['aspiration'] = le.fit_transform(df['aspiration'])

    df['doornumber'] = le.fit_transform(df['doornumber'])

    df['carbody'] = le.fit_transform(df['carbody'])

    df['drivewheel'] = le.fit_transform(df['drivewheel'])

    df['enginelocation'] = le.fit_transform(df['enginelocation'])

    df['enginetype'] = le.fit_transform(df['enginetype'])

    df['cylindernumber'] = le.fit_transform(df['cylindernumber'])

    df['fuelsystem'] = le.fit_transform(df['fuelsystem'])

    #Split the data into features and target variable

    X = df.iloc[:, :-1]

    y = df.iloc[:, -1]

    #Split the dataset into training and testing datasets

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    #Apply feature selection techniques on the training dataset

    #Method 1: Univariate Selection

    selector = SelectKBest(score_func=f_regression, k=10)

    X_train_new = selector.fit_transform(X_train, y_train)

    mask = selector.get_support()

    selected_features = X_train.columns[mask]

    #Method 2: Feature Importance using Extra Trees Regressor

    model = ExtraTreesRegressor()

    model.fit(X_train, y_train)

    importance = model.feature_importances_

    indices = np.argsort(importance)[::-1]

    selected_features = X_train.columns[indices][:10]

    #Save the important features with the target variable to a new CSV file

    df_new = pd.concat([X_train[selected_features], y_train], axis=1)

    df_new.to_csv('CarPrice_new.csv', index=False)

# OUPUT:
 
 ![image](https://user-images.githubusercontent.com/91734840/234252082-749aa0d6-72c0-4e04-894a-178911ccae2f.png)
 
# CODE FOR “titanic_dataset.csv”:

        import pandas as pd

        import numpy as np

        from sklearn.preprocessing import LabelEncoder

        from sklearn.impute import SimpleImputer

        from sklearn.feature_selection import SelectKBest

        from sklearn.feature_selection import chi2

        #Load data

        df = pd.read_csv('titanic_dataset.csv')

        #Drop unnecessary columns

        df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

        #Encode categorical features

        le = LabelEncoder()

        df['Sex'] = le.fit_transform(df['Sex'])

        df['Embarked'] = le.fit_transform(df['Embarked'].astype(str))

        #Impute missing values

        imputer = SimpleImputer(missing_values=np.nan, strategy='median')

        df[['Age']] = imputer.fit_transform(df[['Age']])

        #Perform feature selection

        X = df.iloc[:, :-1]

        y = df.iloc[:, -1]

        selector = SelectKBest(chi2, k=3)

        X_new = selector.fit_transform(X, y)

        #Save transformed data into new file

        df_new = pd.DataFrame(X_new, columns=['Pclass', 'Age', 'Fare'])

        df_new['Survived'] = y.values

        df_new.to_csv('titanic_transformed.csv', index=False)

# OUTPUT:

![image](https://user-images.githubusercontent.com/91734840/234304702-796772b9-1cfe-49b0-bd69-f5aadd74c95d.png)

# RESULT:

Thus, to perform the various feature selection techniques on a dataset and save the data to a file has been performed successfully.
