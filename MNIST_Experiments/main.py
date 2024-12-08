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

def getIndexBasedOnIQR(data, iqr_factor=1.5):
  # Calculate Q1 (25th percentile) and Q3 (75th percentile)
  Q1 = np.percentile(data, 25)
  Q3 = np.percentile(data, 75)

  # Calculate the Interquartile Range (IQR)
  IQR = Q3 - Q1

  # Define the lower and upper bounds for outliers
  lower_bound = Q1 - iqr_factor * IQR
  upper_bound = Q3 + iqr_factor * IQR

  index = (data >= lower_bound) & (data <= upper_bound)
  return index

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor

def check_or_create_directory(path):
    if os.path.exists(path):
        print(f"The directory '{path}' already exists.")
    else:
        os.makedirs(path)
        print(f"The directory '{path}' has been created.")


check_or_create_directory(os.getcwd() + "/Models_ISI/")
check_or_create_directory(os.getcwd() + "/Models_CV/")
check_or_create_directory(os.getcwd() + "/Scalers/")

def getLabelData(metric_type, op_spk_per_cls, features_per_cls):

  op_spk_per_cls = [[idx+1 for idx,spk in enumerate(op_spk) if spk == 1] for op_spk in op_spk_per_cls]

  if metric_type == "isi":
    avg_isi = []
    for op_spk in op_spk_per_cls:
      avg_isi.append(getAvgIsi(op_spk))
    metric = np.array(avg_isi)

  elif metric_type == "cv":
    cv = []
    for op_spk in op_spk_per_cls:
      cv.append(coefficientOfVariation(op_spk))
    metric = np.array(cv)

  non_zero_idx = np.where(metric!=0)[0]
  features_per_cls = features_per_cls[non_zero_idx]
  metric = metric[non_zero_idx]
  index_to_select = getIndexBasedOnIQR(metric, iqr_factor=1.5)

  return metric[index_to_select], features_per_cls[index_to_select]

def run(metric_type, MAEs,  data_path, correct_target_path, n_classes):

  for cls in range(n_classes):

    if cls not in MAEs:
      MAEs[cls] = {}

    print("*"*75)
    print(f"Class: {cls}")
    print("*"*75)
    file_name = "data_mnist_cls_" + str(cls) + ".npy"
    print(f"File Name: {file_name}")
    # Loading the data
    with open(data_path + file_name, 'rb') as f:
      data_per_cls = np.load(f)
    print(f"Data shape: {data_per_cls.shape}")

    # Collecting features
    features_per_cls = []
    for data in data_per_cls:  # (#samples_per_cls, timesteps + padding (100+10), w*h (16*16))
      features_per_time_step = []
      for time_step_data in data:
        features_per_time_step += getFeatures(time_step_data)

      features_per_cls.append(features_per_time_step)
    features_per_cls = np.array(features_per_cls)

    features_per_cls = np.nan_to_num(features_per_cls)
    print(f"Features shape: {features_per_cls.shape}")

    file_name1 = "target_mnist_cls_" + str(cls) + ".npy"
    # Loading their correspnding output spike events
    with open(correct_target_path + file_name1, 'rb') as f:
      op_spk_per_cls = np.load(f)
    print(f"op_spk_per_cls shape: {op_spk_per_cls.shape}")

    metric, features_per_cls =  getLabelData(metric_type, op_spk_per_cls, features_per_cls)

    X_train, X_val, y_train, y_val = train_test_split(features_per_cls, metric, test_size=0.10, random_state=42)
    X_train_norm, X_val_norm, scaler = getNormalizedData("rob", X_train, X_val) # 'std', 'minmax'
    print(f"X_train_norm shape: {X_train_norm.shape}")
    print(f"X_val_norm shape: {X_val_norm.shape}")
    print()

    # Saving scalar
    with open(os.getcwd() + "/Scalers/Scaler_per_cls_" + str(cls) + ".pkl" , 'wb') as scalar_file:
      pickle.dump(scaler, scalar_file)

    # Defining models
    regressors_name = [ "RandomForest", "HistGradientBoosting", "XGB", "LGBM", "CatBoost", "ExtraTrees"]

    # Below hyperparameters of the models are selected by grid search.
    rf =  RandomForestRegressor(n_estimators = 250, criterion='squared_error', max_features = 'sqrt', max_depth=6, min_samples_split=2, random_state=42)
    hist = HistGradientBoostingRegressor(learning_rate=0.01, max_leaf_nodes=31, min_samples_leaf=10, max_iter=120, max_depth=8, random_state=42)
    xgb = XGBRegressor(n_estimators = 100, max_depth = 10, learning_rate=0.01, booster = 'gbtree', tree_method = 'auto',random_state=42)
    lgbm = LGBMRegressor(n_estimators = 200, boosting_type = 'dart',max_depth = 7, num_leaves = 36, learning_rate=0.03, random_state =42, verbose=-1)
    cat = CatBoostRegressor(learning_rate = 0.01,boosting_type = 'Plain',loss_function = "RMSE",verbose = False)
    exrf = ExtraTreesRegressor(n_estimators = 250, criterion='squared_error', max_features = 'log2', max_depth=6, min_samples_split=2, random_state=42)
    regressors = [rf, hist, xgb, lgbm, cat, exrf]

    for idx, regressor in enumerate(regressors):
      print(f"Regressor: {regressors_name[idx]}")
      regressor.fit(X_train_norm, y_train)

      # Prediction
      prediction_val = regressor.predict(X_val_norm)
      prediction_train = regressor.predict(X_train_norm)

      # saving regressor model
      if metric_type == "isi":
        with open(os.getcwd() + "/Models_ISI/" + regressors_name[idx] + "_cls_" + str(cls) + ".pkl" , 'wb') as file:
          pickle.dump(regressor, file)
      else:
        with open(os.getcwd() + "/Models_CV/" + regressors_name[idx] + "_cls_" + str(cls) + ".pkl" , 'wb') as file:
          pickle.dump(regressor, file)

      print("Validation Set")
      mae_val = mean_absolute_error(y_val, prediction_val)
      print("MAE Validation:", mae_val)

      print()
      print("Training Set")
      mae_train = mean_absolute_error(y_train, prediction_train)

      print("MAE Training:", mae_train)
      print("*"*75)
      print("\n")


      if regressors_name[idx] not in MAEs[cls]:
        MAEs[cls][regressors_name[idx]] = [mae_val, mae_train] # [validation MAE, Training MAE]

  if metric_type == "isi":
    np.save(os.getcwd()+"/MAEs_ISI.npy", MAEs)
  elif metric_type == "cv":
    np.save(os.getcwd()+"/MAEs_CV.npy", MAEs)

# For average ISI
data_path = os.getcwd() + "/Data/"
correct_target_path = os.getcwd() + "/Correct_output_spk_per_cls/"

MAEs = {}
metric_type = "isi"
n_classes = 10
MAEs_ISI = {}

run(metric_type, MAEs_ISI,  data_path, correct_target_path, n_classes)

# For CV
data_path = os.getcwd() + "/Data/"
correct_target_path = os.getcwd() + "/Correct_output_spk_per_cls/"

MAEs = {}
metric_type = "cv"
n_classes = 10
MAEs_CV = {}

run(metric_type, MAEs_CV,  data_path, correct_target_path, n_classes)



