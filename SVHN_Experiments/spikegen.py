import torch, os
import numpy as np
import snntorch as snn
from snntorch import spikegen
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def process_aer(aer,pad_samples=10):
    aer_data = aer.detach().cpu().numpy()
    (time_steps,n_images,n_input) = aer_data.shape

    aer_pad = np.zeros((pad_samples,n_input))
    aer_out = np.zeros((pad_samples,n_input))

    for i in range(n_images):
        aer_out = np.concatenate((aer_out,aer_data[:,i,:],aer_pad),axis=0)

    return aer_out

def flush_aer(aer,pad_samples=10,all_dump=True):
    aer_in = process_aer(aer,pad_samples)
    aer_hw = np.flip(aer_in,axis=1)      # flip input: verilog uses [(n-1):0] while any data structure in python uses [0:(n-1)]
    return aer_hw

transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((16, 16)),
                transforms.Normalize((0,), (1,))
                ])

train_data = datasets.MNIST(os.getcwd(), train=True, download=True, transform=transform)
test_data = datasets.MNIST(os.getcwd(), train=False, download=True, transform=transform)

batch_size = 1000

torch.manual_seed(42)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)

def getSpikeGenData(loader, num_steps=100, pad_samples=10, path_per_type='', target_path= '', type_=''):
  strt_idx = 0
  for data, targets in loader:
    batch = data.shape[0]
    sample_aer = spikegen.rate(data.view(batch,-1),num_steps=num_steps)
    aer_hw = flush_aer(sample_aer,pad_samples=pad_samples)
    np.savetxt(path_per_type+"Spk_"+type_+"_img_" + str(strt_idx) +"_" + str(strt_idx + batch) + ".txt" , aer_hw,delimiter='',fmt='%d')
    np.savetxt(target_path+"Spk_"+ type_+"_label_" + str(strt_idx) +"_" + str(strt_idx + batch) + ".txt" , targets.numpy(), delimiter='',fmt='%d')
    print(f"Start_index: {strt_idx} | End_idx: {strt_idx + batch}")
    strt_idx += batch

train_spk_path = "SpikeGenData_train_svhn/"
test_spk_path = "SpikeGenData_test_svhn/"
train_target_path = "target_train_svhn/"
test_target_path = "target_test_svhn/"

def check_or_create_directory(path):
    if os.path.exists(path):
        print(f"The directory '{path}' already exists.")
    else:
        os.makedirs(path)
        print(f"The directory '{path}' has been created.")

check_or_create_directory(train_spk_path)
check_or_create_directory(test_spk_path)
check_or_create_directory(train_target_path)
check_or_create_directory(test_target_path)

getSpikeGenData(train_loader, num_steps=100, pad_samples=10, path_per_type=train_spk_path, target_path= train_target_path, type_='train')
getSpikeGenData(test_loader, num_steps=100, pad_samples=10, path_per_type=test_spk_path, target_path= test_target_path, type_='test')



