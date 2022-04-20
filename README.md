# RotNet: A MEC Analysis

## Code Structure
### MEC RotNet
1. [1_GetFeatures.ipynb](1_GetFeatures.ipynb) will train an autoencoder for CIFAR10 to generate features for CIFAR10. This will generate a CSV of data (features, {0 if image not rotated, 1 o.w.})
2. [2_TrainBrainome.ipynb](2_TrainBrainome.ipynb) will train the MEC sized MLP for rotation prediction. 
3. [3_Eval.ipynb](3_Eval.ipynb) will evaluate the MLP features from step 2 on CIFAR10 using linear regression.

### RotNet Baseline
We have also retrained the original RotNet for the task of binary rotation prediction. See [baselines](./baselines/) for code and instructions.


