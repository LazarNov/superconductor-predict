# superconductor-predict

Hello. This is the code for prediction of superconducting transition temperatures, $T_c$, of structures. This work can be found at https://arxiv.org/abs/2301.10474 and uses the code from https://github.com/khamidieh/predict_tc to add stoichiometric features as well.
Making this accessible is the goal, feel free to contact me about anything  

# Using the model
Use Predicting.ipynb to predict superconducting temperature of a file


First install r packages CHNOSZ and xgboost if necessary.

```python
import rpy2.robjects
from rpy2.robjects import packages
utils = packages.importr('utils')
base =  packages.importr('base')

utils.install_packages("CHNOSZ")
utils.install_packages("xgboost")
     
```

Load the model.

```python

from slpred import slpred

ms=slpred.FourierSt(pr=True)
modelms=ms.samplemodels(modeltype='predictivesavepr',typea='original')
     
```

List the structures and predict 
.

```python

structs=["slpred/slpred/C1S1H7.cif"]
ms.predict_structure(modelms,structs)
```     

```100%|██████████| 1/1 [00:00<00:00, 21.03it/s]
100%|██████████| 1/1 [00:00<00:00, 580.13it/s]
(1, 13, 13, 13, 2) (1, 175)
1/1 [==============================] - 0s 21ms/step
H7CS structure: 1 Tc:[173.69496]
array([[173.69496]], dtype=float32)
```
