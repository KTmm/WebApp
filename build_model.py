from transformer import Cost_Transformer
from model import FraudModel
import pickle

model = FraudModel()
model.train()
with open('models/model.pkl', 'wb') as f:
	pickle.dump(model,f)