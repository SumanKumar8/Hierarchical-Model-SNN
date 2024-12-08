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

# Output spikes obtained from QUANTISENC after passing the data generated from executing spikegen.py
output_files = os.listdir('Output_spike_train_Fashion_mnist/')
output_files = [x for x in output_files if x.endswith('.txt')]
output_files.sort(key=natural_keys)


target_files = os.listdir('target_train_Fashion_mnist/')
target_files = [x for x in target_files if x.endswith('.txt')]
target_files.sort(key=natural_keys)


# To verfify how much performance (in terms of accuracy), we are getting from QUANTISENC output
hw_pred = []
target = []

for op_file, tr_file in zip(output_files, target_files[:len(output_files)]):
    print(op_file, "|", tr_file)
    hw_preds = process_spikes('Output_spike_train_Fashion_mnist/' + op_file,reverse=True, n_classes = n_classes).numpy().tolist()
    hw_pred += hw_preds
    gt = open('target_train_Fashion_mnist/' + tr_file).readlines()
    targets = [int(x.strip()) for x in gt]
    target += targets
    print("Accuracy:", (np.array(hw_preds) == np.array(targets)).sum()/ len(targets))
    print()

"""
Collecting the correct indices of the examples so that later, we can build machine learning
models trained only on correcly classified examples.
"""
data_correct_indices = []
data_incorrect_indices = []

for idx, labels in enumerate(zip(hw_pred, target)):
    p, t = labels
    if p == t:
        data_correct_indices.append(idx)
    else:
        data_incorrect_indices.append(idx)

input_files = os.listdir('SpikeGenData_train_Fashion_mnist/')
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
    ip_file = open('SpikeGenData_train_Fashion_mnist/' + ip_file, 'r').readlines()
    ip_file = [[int(e) for e in x.strip()] for x in ip_file[:-10]]
    print("Number of samples:", len(ip_file))
    print()
    for i in range(0, len(ip_file), pad_time_step):
      dataset.append(ip_file[i:i+pad_time_step])

print()
print("Total number of samples:", len(dataset))

# For ouput spikes from QUANTISENC
output_label = []

for op_file in output_files:
    print(op_file)
    hw_op = open('Output_spike_train_Fashion_mnist/' + op_file, 'r').readlines()
    hw_op = [[int(e) for e in x.strip()] for x in hw_op[15:-15]] # Discarding the first and last 15 time steps
    print("Number of samples:", len(hw_op))
    print()
    for i in range(0, len(hw_op), pad_time_step):
      output_label.append(hw_op[i:i+pad_time_step])

print()

dataset_v1 = np.array(dataset)
output_label_v1 = np.array(output_label)

del dataset, output_label

gc.collect()

print(f"Total number of correcly classified examples: {len(data_correct_indices)}.")

target = np.array(target)

# Selecting only correctly classified examples
correct_target = target[data_correct_indices]

classes, counts = np.unique(correct_target, return_counts=True)

def check_or_create_directory(path):
    if os.path.exists(path):
        print(f"The directory '{path}' already exists.")
    else:
        os.makedirs(path)
        print(f"The directory '{path}' has been created.")

data_path = os.getcwd() + "/Data/"
check_or_create_directory(data_path)
print()
correct_target_path = os.getcwd() + "/Correct_output_spk_per_cls/"
check_or_create_directory(correct_target_path)

for cls in classes:
  print(f"Class: {cls}")
  # Selecting correctly classified index for each class
  index_correct_target_per_cls = np.where(correct_target == cls)[0]
  print(f"Number of samples for class: {index_correct_target_per_cls.shape[0]}")
  file_name = "data_Fashion_mnist_cls_" + str(cls) + ".npy"
  print(f"File Name: {file_name}")

  # Saving the data
  with open(data_path + file_name, 'wb') as f:
    np.save(f, dataset_v1[data_correct_indices][index_correct_target_per_cls])

  file_name1 = "target_Fashion_mnist_cls_" + str(cls) + ".npy"

  target_per_cls = np.zeros((index_correct_target_per_cls.shape[0], pad_time_step))

  """
  Fetching output spikes based on ground truth and select the correct class from right
  to left because Vivado deals in that way.
  """
  for idx, op_spk in enumerate(output_label_v1[data_correct_indices][index_correct_target_per_cls]):
    target_per_cls[idx] = op_spk[:, (n_classes - 1) - cls] # (9-cls)

  # Saving the label
  with open(correct_target_path + file_name1, 'wb') as f:
    np.save(f, target_per_cls)

  print()



