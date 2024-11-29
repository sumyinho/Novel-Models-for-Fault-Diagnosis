from util.utils import load_CWRU_data
import pickle

root = "../dataset/CWRU/"

data = load_CWRU_data(gap=800)
# with open(root + "CWRU.pkl", 'wb') as f:
#     pickle.dump(data, f)
#     print("Write data completed!!!")
