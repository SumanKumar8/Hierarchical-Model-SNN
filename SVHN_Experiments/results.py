import numpy as np
import os, re, torch, gc

def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):

    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]

def process_spikes(ofname,reverse=False, n_classes=10):
    file    = open(ofname,'r')
    lines   = file.readlines()
    hwout   = list()
    for line  in lines:
        out = [int(x) for x in list(line.strip())]
        if reverse:
            hwout.append(out[::-1])
        else:
            hwout.append(out)
    hardware_skips_start = 15
    hardware_skip_end    = 15

    hwout_processed = hwout[hardware_skips_start:-hardware_skip_end]
    total_samples  = len(hwout_processed)
    n_output = len(hwout_processed[0])

    n_images = total_samples / (time_steps + pad_samples)
    hwout_processed_np = np.zeros((total_samples,n_output))
    for i,oevent in enumerate(hwout_processed):
        hwout_processed_np[i] = np.array(oevent)

    hwout_labels = list()
    start_index = 0
    for i in range(int(n_images)):
        end_index = start_index + time_steps + pad_samples
        output = torch.tensor(hwout_processed_np[start_index:end_index].reshape(-1,1, n_classes))
        _, idx    = output.sum(dim=0).max(1)
        hwout_labels.append(idx)
        start_index = end_index

    return torch.stack(hwout_labels).view(-1)

pad_samples, time_steps, n_classes = 10, 100, 10

# Output spikes (either missing or delayed or spurious) collected from QUANTISENC after injecting fault at fault sites: FS1, 2, and 3.
output_files = os.listdir('Output_spike_test_inject_svhn/')
output_files = [x for x in output_files if x.endswith('.txt')]
output_files.sort(key=natural_keys)

input_files = os.listdir('SpikeGenData_test_svhn/')
input_files = [x for x in input_files if x.endswith('.txt')]
input_files.sort(key=natural_keys)

"""
batch_size = 1000
num_time_steps = 100
padding (P) = 10
I: Image

ip = [P, I, P, I, ...., P] # 110,010 --> Discard the last 10 time steps --> 110,000

#op 110,030 --> Discard the first and last 15 time steps --> 110,00

"""
dataset = []
pad_time_step = 10 + 100
for ip_file in input_files[:len(output_files)]:
    print("File:", ip_file)
    ip_file = open('SpikeGenData_test_svhn/' + ip_file, 'r').readlines()
    ip_file = [[int(e) for e in x.strip()] for x in ip_file[:-10]]
    print("Number of samples:", len(ip_file))
    print()
    for i in range(0, len(ip_file), pad_time_step):
      dataset.append(ip_file[i:i+pad_time_step])

print()
print("Total number of samples:", len(dataset))

# Output spikes (either missing or delayed or spurious) collected from QUANTISENC after injecting fault at fault sites: FS1, 2, and 3.
output_label = []

for op_file in output_files:
    print(op_file)
    hw_op = open('Output_spike_test_inject_svhn/' + op_file, 'r').readlines()
    hw_op = [[int(e) for e in x.strip()] for x in hw_op[15:-15]] # Discarding the first and last 15 time steps
    print("Number of samples:", len(hw_op))
    print()
    for i in range(0, len(hw_op), pad_time_step):
      output_label.append(hw_op[i:i+pad_time_step])

print()
print("Total number of labels:", len(output_label))

dataset_v1 = np.array(dataset)
output_label_v1 = np.array(output_label)

del dataset, output_label

gc.collect()

classes = list(range(10))
n_classes = len(classes)

MAEs_ISI = np.load(os.getcwd()+"/MAEs_ISI.npy", allow_pickle=True).tolist()
MAEs_CV = np.load(os.getcwd()+"/MAEs_CV.npy", allow_pickle=True).tolist()

import os, pickle
import numpy as np
import pandas as pd
from scipy.fft import fft
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

def getNormalizedData(Type, X_train, X_val):
  """
  args:
  Type: Minmax or Standard scaling technique
  X_train: Training data
  X_val: Validation data

  return X_train_norm, X_val_norm, X_test_norm # Normalized
  """

  if Type == "minmax":
    print("Using MinMaxScaler")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X_train)
  elif Type == "std":
    print("Using StandardScaler")
    scaler = StandardScaler()
    scaler.fit(X_train)
  else:
    print("Using RobustScaler")
    scaler = RobustScaler()
    scaler.fit(X_train)

  X_train_norm, X_val_norm = scaler.transform(X_train), scaler.transform(X_val)

  # Replace NaN values with zero
  return np.nan_to_num(X_train_norm), np.nan_to_num(X_val_norm), scaler

# To compute average ISI (Inter-spike interval)
def getAvgIsi(x):

  if len(x) == 0:
    return 0

  elif len(x) == 1:
    # return x[0]%1024
    return x[0]

  else:
    # x = [i%1024 for i in x]
    x = [i for i in x]
    if x[0] == 0:
      x[0] = 1
    return sum([(x[i+1])- (x[i]) for i in range(len(x)-1)])/(len(x)-1)



# To compute COV (coefficient of variation)
def coefficientOfVariation(x):

  if len(x) == 0:
    return 0

  elif len(x) == 1:
    # return x[0]%1024
    return x[0]

  else:
    # x = [i%1024 for i in x]
    x = [i for i in x]
    if x[0] == 0:
      x[0] = 1
    imd = np.array([x[i] for i in range(len(x))])
    mu = np.mean(imd)

    # coefficient of variation
    cov = np.sqrt(np.sum(np.square(imd - mu))/ (len(imd) - 1))/mu

    return cov


def detect_bursts_and_compute_features(spike_times, min_spikes=3, isi_threshold=0.1):
    """
    Detect bursts and compute burst-related features from spike times.

    Parameters:
    - spike_times: array-like, spike times in seconds.
    - min_spikes: minimum number of spikes in a burst.
    - isi_threshold: maximum inter-spike interval (ISI) in seconds to consider spikes part of the same burst.

    Returns:
    - features: dict containing burst durations, inter-burst intervals, and burst frequency.
    """
    isis = np.diff(spike_times)  # Calculate inter-spike intervals
    bursts = []
    burst_start_indices = []

    for i, isi in enumerate(isis):
        if isi <= isi_threshold:
            if len(burst_start_indices) == 0:
                burst_start_indices.append(i)  # Start of a new burst
        else:
            if len(burst_start_indices) > 0 and (i - burst_start_indices[-1] + 1) >= min_spikes:
                # End of a burst
                bursts.append((spike_times[burst_start_indices[-1]], spike_times[i + 1]))
                burst_start_indices = []
            else:
                burst_start_indices = []
    # Check for a burst at the end
    if len(burst_start_indices) > 0 and (len(spike_times) - burst_start_indices[0]) >= min_spikes:
        bursts.append((spike_times[burst_start_indices[0]], spike_times[-1]))

    # Calculate features
    burst_durations = [end - start for start, end in bursts]
    inter_burst_intervals = [bursts[i + 1][0] - bursts[i][1] for i in range(len(bursts) - 1)]
    burst_frequency = len(bursts) / (spike_times[-1] - spike_times[0]) if bursts else 0

    # return [np.mean(burst_durations), np.median(burst_durations), np.std(burst_durations), np.mean(inter_burst_intervals), np.median(inter_burst_intervals), np.std(inter_burst_intervals), burst_frequency]
    return burst_frequency


def compute_fano_factor(spike_times, interval_length=64):
    max_time = spike_times[-1]
    intervals = np.arange(0, max_time, interval_length)
    counts = np.histogram(spike_times, bins=intervals)[0]
    ff = np.var(counts) / np.mean(counts)
    return ff

def getBinary(y_true, y_pred, margin):

  """
  It will binarize the continuous values which are y_true and y_pred with
  the help of margin and threshold to plot ROC (Receiver operating characteristic).

  Args:
  y_true : Observed avg. ISI (continuous values)
  y_pred : Predicted avg. ISI (continuous values)
  margin : The margin is the obtained optimal MAE or a user can take loose margin as per his choice to incur more false alaram or false fault

  return:
  y_t : Binarized y_true
  y_p : Binarized y_pred

  The reason to opt for ROC curve because it offers an elegant way to
  plot true fault detection rate versus false fault detection rate.

  """

  thresh = pd.Series(y_true).median()   # We have taken median as a threshold because it is not impacted by the outliers
  y_t = []
  y_p = []

  for t,p in zip(y_true, y_pred):
    if np.abs(t - p) <= margin:
      if t> thresh:
        y_t.append(1)
        y_p.append(1)
      else:
        y_t.append(0)
        y_p.append(0)

    else:
      if (t > thresh) and (p > thresh):  # ex: t=60, p=80 but thresh = 55 --> t,p -> 1, therefore one of them has to be opposite to another because abs(t-p)>15.
        y_t.append(1)
        y_p.append(0)

      elif (t > thresh) and (p <= thresh):  # ex: t=60, p=40 but thresh = 55 --> t->1,  p-> 0, which satisfies the condition, one has to be opposite of another.
        y_t.append(1)
        y_p.append(0)

      elif (t <= thresh) and (p > thresh):  # ex: t=30, p=57 but thresh = 55 --> t->0,  p-> 1,
        y_t.append(0)
        y_p.append(1)

      elif (t <= thresh) and (p <= thresh): # ex: t=25, p=45 but thresh = 55 --> t->0, p-> 1, therefore one of them has to be opposite to another.
        y_t.append(0)
        y_p.append(1)


  y_t = np.array(y_t)
  y_p = np.array(y_p)

  return y_t, y_p

def getFeatures(spk_event):
  if sum(spk_event) == 0:
    return [0, 0, 0, 0, 0]

  spk_event = [idx+1 for idx,spk in enumerate(spk_event) if spk == 1]

  avg_isi = getAvgIsi(spk_event)
  cov = coefficientOfVariation(spk_event)
  burst_frequency = detect_bursts_and_compute_features(spk_event)
  fano = compute_fano_factor(spk_event)
  numSpikes = len(spk_event)
  return [avg_isi, cov, burst_frequency, fano, numSpikes]

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor

# Collecting features
features = []
for data in dataset_v1:  # (#samples_per_cls, timesteps + padding (100+10), w*h (16*16))
  features_per_time_step = []
  for time_step_data in data:
    features_per_time_step += getFeatures(time_step_data)

  features.append(features_per_time_step)
features = np.array(features)
print(f"Features shape: {features.shape}")

features = np.nan_to_num(features)

metric_type = "isi"
regressors_name = [ "RandomForest", "HistGradientBoosting", "XGB", "LGBM", "CatBoost", "ExtraTrees"]

for index,model_name in enumerate(regressors_name):
  prediction_per_neuron_cls = []
  for cls in classes:
      print("Neuron or Class:", cls, end ="|")
      # Loading scalar
      with open(os.getcwd() + "/Scalers/Scaler_per_cls_" + str(cls) + ".pkl" , 'rb') as scalar_file:
        scaler = pickle.load(scalar_file)

      # Normalizing the features
      features_norm = scaler.transform(features)

      if metric_type == "isi":
        mae_val, mae_train = MAEs_ISI[cls][model_name]
        with open(os.getcwd() + "/Models_ISI/" + model_name + "_cls_" + str(cls) + ".pkl", 'rb') as file:
          regressor = pickle.load(file)

      elif metric_type == "cv":
        mae_val, mae_train = MAEs_CV[cls][model_name]
        with open(os.getcwd() + "/Models_CV/" + model_name + "_cls_" + str(cls) + ".pkl", 'rb') as file:
          regressor = pickle.load(file)

      print("Regressor:", model_name)
      # Predicting
      y_pred = regressor.predict(features_norm)
      margin = mae_val
      # margin = mae_train

      inject_target_per_cls = np.zeros((dataset_v1.shape[0], pad_time_step))
      for idx, op_spk in enumerate(output_label_v1):
        inject_target_per_cls[idx] = op_spk[:, (n_classes - 1) - cls] # (9-cls)

      eval_with_fault = [[idx+1 for idx,spk in enumerate(inject_spk) if spk == 1] for inject_spk in inject_target_per_cls]

      if metric_type == "isi":
        eval_with_fault_avg_ISI = [getAvgIsi(spk_event) for spk_event in eval_with_fault]
        y_t, y_p = getBinary(eval_with_fault_avg_ISI, y_pred, margin)
      elif metric_type == "cv":
        eval_with_fault_CV = [coefficientOfVariation(spk_event) for spk_event in eval_with_fault]
        y_t, y_p = getBinary(eval_with_fault_CV, y_pred, margin)

      prediction_per_neuron_cls.append((y_t!= y_p).tolist())


  prediction_per_neuron_cls = np.array( prediction_per_neuron_cls)
  tps = []
  for cls in classes:
    tp = np.sum(prediction_per_neuron_cls[cls])/ len(prediction_per_neuron_cls[cls])
    tps.append(tp)

  print()
  print("Mean:", np.mean(tps))
  print()
  print("*"*75)

metric_type = "cv"
regressors_name = [ "RandomForest", "HistGradientBoosting", "XGB", "LGBM", "CatBoost", "ExtraTrees"]

for index,model_name in enumerate(regressors_name):
  prediction_per_neuron_cls = []
  for cls in classes:
      print("Neuron or Class:", cls, end ="|")
      # Loading scalar
      with open(os.getcwd() + "/Scalers/Scaler_per_cls_" + str(cls) + ".pkl" , 'rb') as scalar_file:
        scaler = pickle.load(scalar_file)

      # Normalizing the features
      features_norm = scaler.transform(features)

      if metric_type == "isi":
        mae_val, mae_train = MAEs_ISI[cls][model_name]
        with open(os.getcwd() + "/Models_ISI/" + model_name + "_cls_" + str(cls) + ".pkl", 'rb') as file:
          regressor = pickle.load(file)

      elif metric_type == "cv":
        mae_val, mae_train = MAEs_CV[cls][model_name]
        with open(os.getcwd() + "/Models_CV/" + model_name + "_cls_" + str(cls) + ".pkl", 'rb') as file:
          regressor = pickle.load(file)

      print("Regressor:", model_name)
      # Predicting
      y_pred = regressor.predict(features_norm)
      np.random.seed(42)

      margin = mae_val
      # margin = mae_train

      inject_target_per_cls = np.zeros((dataset_v1.shape[0], pad_time_step))
      for idx, op_spk in enumerate(output_label_v1):
        inject_target_per_cls[idx] = op_spk[:, (n_classes - 1) - cls] # (9-cls)

      eval_with_fault = [[idx+1 for idx,spk in enumerate(inject_spk) if spk == 1] for inject_spk in inject_target_per_cls]

      if metric_type == "isi":
        eval_with_fault_avg_ISI = [getAvgIsi(spk_event) for spk_event in eval_with_fault]
        y_t, y_p = getBinary(eval_with_fault_avg_ISI, y_pred, margin)
      elif metric_type == "cv":
        eval_with_fault_CV = [coefficientOfVariation(spk_event) for spk_event in eval_with_fault]
        y_t, y_p = getBinary(eval_with_fault_CV, y_pred, margin)

      prediction_per_neuron_cls.append((y_t!= y_p).tolist())


  prediction_per_neuron_cls = np.array( prediction_per_neuron_cls)
  tps = []
  for cls in classes:
    tp = np.sum(prediction_per_neuron_cls[cls])/ len(prediction_per_neuron_cls[cls])
    tps.append(tp)

  print()
  print("Mean:", np.mean(tps))
  print()
  print("*"*75)

for reg in regressors_name:
  print(reg)
  mae_isi_reg = []
  mae_cv_reg = []
  for cls in classes:
    mae_isi_reg.append(MAEs_ISI[cls][reg][0])
    mae_cv_reg.append(MAEs_CV[cls][reg][0])

  print("ISI:")
  print(mae_isi_reg)
  print("Mean:", np.mean(mae_isi_reg))
  print("Var:",np.var(mae_isi_reg))
  print()
  print("CV:")
  print(mae_cv_reg)
  print("Mean:", np.mean(mae_cv_reg))
  print("Var:", np.var(mae_cv_reg))
  print("*"*75)



