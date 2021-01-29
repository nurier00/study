import pandas as pd
from sklearn.datasets import load_breast_cancer


def load_data():
    cancer = load_breast_cancer()

    df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    df['class'] = cancer.target

    print(df.tail())
    print(f"columns : {df.columns}")

    return df
