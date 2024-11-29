from util.utils import load_PU_data
import pickle

root = "../dataset/Paderborn_manual_damage/"

data = load_PU_data(gap=430)
# with open(root + "PU.pkl", 'wb') as f:
#     pickle.dump(data, f)
#     print("Write data completed!!!")
