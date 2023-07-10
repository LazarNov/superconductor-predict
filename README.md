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



## Saved models

```python
ms=FourierSt(pr=True)
modelms=ms.samplemodels(modeltype='predictivesavepr',typea='original')

```

``modeltype='predictivesavepr'`` for saved predictive model with more accuracte gaussians. 
typea refers to the density components coefficients. 
``type='original'`` is used in the saved models and uses complex numbers for each spatial characteristic,
1 for mass and 1 for charge in this case.


## Predict with saved models 

#### For predicting $T_c$ from a list [cifs] 

```python
ms.predict_structure(modelms,['C1S1H7.cif'])

```

#### For predicting $T_c$ from a list of pymatgen structures [pmgs] 

``ms.data_remove_duplicates().structures[0:10]`` are the first 10 from the default structures saved.

```python
ms.predict_structure(modelms,ms.data_remove_duplicates().structures[0:10],structureFiles=False)

```

## Optimizing the structures

#### Optimizing for maximum $T_c$

```python
ms.superconductoropt(modelms,['C1S1H7.cif'])

```

#### Optimizing for picked $T_c$

```python
ms.superconductoropt(modelms,['C1S1H7.cif'],optmax=False,target=180)

```

#### Specify bounds and optimization latparamaters angles


```python
ms.superconductoropt(modelms,['C1S1H7.cif'],lbounds=[[2,12],[2,12],[2,12],[60,120],[60,120],[60,120]],option={'method':'nelder-mead','tol':0.1,'options':{'maxfev':3000}})

```

#### Show last optimization
```python
ms.last_opt

```

#### Start optimization from last structure

```python
minimaobj=ms.superconductoropt(modelms,[ms.last_opt],structuresfiles=False)

```

#### Make cif and crystal with Z= specifying atomic numbers 

```python
makeAndWriteCrystal(minimaobj.x,fileName="optout.cif",buffer=[6,1,1,1,1,1,1,1,16])

```

#### Make cif from last optimization

```python
pymatgen.io.cif.CifWriter(ms.last_opt['structure']).write_file('optout.cif')

```

## Generate and fit sample model and data

#### Returns trained model and data
```python
F=FourierSt(pr=True)
modeli,outputs=F.buildsample(modeltype='predictive',typea='original')
```

#### Build sample data

```python
F=FourierSt(pr=True)
outputs=F.sampledata(typea='original')
```

#### Build and fit prebuilt model or build sample data 

```python
modesl=F.samplemodels(modeltype='predictive',typea='original')
modesl=F.modelfromsample(modesl)
```
