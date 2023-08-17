# AAAI-2023-DeFL
Here, we provide fundamental algorithm implementations within a security context. These simulators encompass the components detailed in our research paper.

# Usage
1. Requirement: Ubuntu 20.04, Python v3.5+, Pytorch and CUDA environment
2. "./Main.py" is about configurations and the basic Federated Learning framework
3. "./Sims.py" describes the simulators for clients and central server
4. "./Attacks.py" gives the codes about attacking methods
5. "./Aggregations.py" shows the realizations of aggregation rules
6. "./Utils.py" contains all necessary functions and discuss how to get training and testing data
7. Folder "./Models" includes codes for AlexNet, FC and VGG-11
8. Folder "./CompFIM" is the package used to compute Fisher Information Matrix (FIM)

# Implementation
 1. Should use "./Main.py" to run results, the command is '''python3 ./Main.py'''
 2. Parameters can be configured in "./Main.py", the main parameters are:
```
 Configs["alpha"] = 0.5
 Configs["attack"] = "MinMax"
 Configs["aggmethod"] = "AFA"
 Configs["attkrate"] = 0.125 # 12.5%
 Configs["learning_rate"] = 0.01
 Configs["wdecay"] = 1e-5
 Configs["batch_size"] = 16
 Configs["iters"] = 200
```

# Citation
If you use the simulator or some results in our paper for a published project, please cite our work by using the following bibtex entry

```
@inproceedings{yan2023defl,
  title={DeFL: Defending Against Model Poisoning Attacks in Federated Learning via Critical Learning Periods Awareness},
  author={Gang Yan, Hao Wang, Xu Yuan and Jian Li},
  booktitle={Proc. of AAAI},
  year={2023}
}
```
