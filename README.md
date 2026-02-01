# FedDep
Public Code of FedDep 
Built upon the PFLlib framework.
This repository provides the implementation of **FedDep**, proposed in:

> Efficient Personalized Federated Learning through Layer-wise Selection and Communication-Aware Optimization

## Requirements
- Python >= 3.8
- PyTorch >= 1.10

```bash
cd ./system
python main.py -data MNIST -m CNN -algo FedDep -gr 2000 -did 0 # using the MNIST dataset, the FedDep algorithm, and the 4-layer CNN model
python main.py -data MNIST -m CNN -algo FedDep -gr 2000 -did 0,1,2,3 # running on multiple GPUs
# FedDep (Patch for PFLlib)

This repository contains **only the necessary modifications** for reproducing **FedDep** and **FedDep-CA** proposed in:

> **FedDep：Personalized Federated Learning through Layer-wise Selection and Communication-Aware Optimization**  
> (submitted to *Computing and Informatics*)

⚠️ **Important:** This repo is **not a standalone runnable codebase**.  
To run the experiments, please **integrate the provided modified files into the original PFLlib framework**.

---

## 1) Prerequisites

- Install and set up **PFLlib** (the official upstream framework) following its original instructions.
- Ensure the Python/PyTorch versions match the PFLlib requirements.

---

## 2) What This Repo Contains

This repo provides:
- The implementation of **FedDep** and **FedDep-CA**
- The required scheduling / layer-wise personalization logic
- The communication-aware sparsification component
- Minimal changes to the training/algorithm entry points (if applicable)

It does **not** include:
- The full PFLlib source code
- Datasets
- All baseline implementations (these are provided by PFLlib)

---

## 3) How to Integrate into PFLlib

### Step 1: Download PFLlib
Clone or download PFLlib from its official repository and complete its environment setup.

### Step 2: Copy the Modified Files
Copy the files/folders from this repo into the corresponding locations in your local PFLlib directory.

A typical integration procedure is:
1. **Overwrite** the matching files in PFLlib with the modified versions from this repo, or
2. **Add** the new algorithm files (e.g., `feddep.py`) into PFLlib’s algorithm module, and
3. Update the algorithm registry / entry file in PFLlib (if needed) so that `feddep` can be selected.

> Please refer to the folder structure in this repo; paths are kept consistent with PFLlib to make integration straightforward.

---

## 4) Running FedDep 

After integration, run experiments using the standard PFLlib command format.

Example (illustrative):
```bash
python main.py --alg feddep --dataset <DATASET_NAME> --model <MODEL_NAME> [other PFLlib args]
