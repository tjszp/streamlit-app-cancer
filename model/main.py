import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle5 as pickle

def get_clean_data():
    data = pd.read_csv('data/cancer-diagnosis.csv')
    data = data.drop(['Unnamed: 32', 'id'], axis=1) # drop unnecessary columns
    data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0}) # map diagnosis to 1 and 0 for malignant and benign
    return data

def create_model(data):
    X = data.drop(['diagnosis'], axis=1) # drop diagnosis column
    y = data['diagnosis'] # set diagnosis as target variable
    
    # standardize data
    scaler = StandardScaler() # scale data
    X_scaled = scaler.fit_transform(X) # fit and transform data
    
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42
    )

    # create and train model
    model = LogisticRegression() # create model
    model.fit(X_train, y_train) # fit model
    
    # print accuracy score - test model
    y_pred = model.predict(X_test) # predict on test data
    print('Accuracy of our model: ', accuracy_score(y_test, y_pred)) # print accuracy score
    print('Classification report: ', classification_report(y_test, y_pred)) # print classification report
    
    return model, scaler


def main():
    data = get_clean_data()
    model, scaler = create_model(data)

    with open('model/model.pkl', 'wb') as file:
        pickle.dump(model, file)
    with open('model/scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)
    
if __name__ == "__main__":
    main()