# Hierarchical-Model-Based-Approach
# DSN2025: Hierarchical Model-Based Approach for Concurrent Testing of Neuromorphic Architecture

## Envrionment

* Python 3.10.12
* Numpy 1.23.5
* Torch 2.1.0+cu118
* Sklearn 1.2.2
* Vivado ML Edition - 2023.1
* Verilog
* System verilog
* Tcl

## To run

### To install the necessary modules

```
pip install -r requirements.txt
```

### To generate spikes from MNIST or FashionMNIST datasets that will feed to QUANTISENC
```
python spikegen.py
```

### To preprocess data to build ML models
```
python preprocess.py
```

### To train tree-based ML models
```
python main.py
```

### To obtain results
```
python results.py
```

## Citation

If you find our work useful, please cite our paper.
```
@inproceedings{kumar2025hierarchical,
  title={Hierarchical Model-Based Approach for Concurrent Testing of Neuromorphic Architecture},
  author={Kumar, Suman and Mishra, Abhishek and Das, Anup and Kandasamy, Nagarajan},
  booktitle={2025 55th Annual IEEE/IFIP International Conference on Dependable Systems and Networks (DSN)},
  pages={511--523},
  year={2025},
  organization={IEEE}
}
```
