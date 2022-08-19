import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from transformer import Cost_Transformer


class FraudModel(object):
    def __init__(self):
        pass

    def build_model(self):
        """
        Building model pipeline
        """
        steps = [('rescale', MinMaxScaler()),
                 ('logr', LogisticRegression(class_weight = 'balanced',random_state=42, C=0.1))]
        self.pipeline = Pipeline(steps)

    def train(self):
        """
        Train a model
        """
        X_train= pd.read_csv('data/train_all_features.csv')
        X_train.fillna(0, inplace=True)
        #
        features = list(X_train.columns)
        #
        # # Define feature and target
        # target = ["price", "luxury"]
        # self.features = [fea for fea in features if fea not in target]


        y_train = pd.read_csv('data/y_train_all_features.csv')
        self.build_model()
        self.model = self.pipeline.fit(X_train, y_train)

    def predict(self, context):
        """
        context: dictionary format {'cost':'$300k'... etc}
        return np.array
        """
        num_predictions = len(context[self.features[0]])
        X = pd.DataFrame(context, index=range(num_predictions))
        return self.model.predict_proba(X)