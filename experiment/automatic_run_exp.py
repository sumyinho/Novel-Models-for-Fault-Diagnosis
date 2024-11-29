import subprocess

# subprocess.run(["python", "exp_ma1dcnn.py"], check=True)
# subprocess.run(["python", "exp_rsbu_new.py"], check=True)
from util.utils import automatic_run_experiments

file_name_list = [
                  "exp_baseline_cnn.py",
                  "exp_baseline_lstm.py",
                  "exp_att_mbigru.py",
                  "exp_mixcnn.py",
                  "exp_dsl.py",
                  "exp_laplace_alexnet.py",
                  "exp_ma1dcnn.py",
                  "exp_rsbu.py",
                  "exp_secn.py"]
automatic_run_experiments(file_name_list)
