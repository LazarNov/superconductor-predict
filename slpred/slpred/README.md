# Using the model
Use Predicting.ipynb to predict superconducting temperature of a file







## Saved models

```python
ms=FourierSt(pr=True)
modelms=ms.samplemodels(modeltype='predictivesavepr',typea='original')

```

``modeltype='predictivesavepr'`` for saved predictive model with more accuracte gaussians. 
typea refers to the density components coefficients. 
``type='original'`` is used in the saved models and uses complex numbers for each spatial charactersitic modeled,
1 for mass and 1 for charge in this case.


## Predict with saved models 

#### For predicting Tc from a list [cifs] 

```python
ms.predict_structure(modelms,['C1S1H7.cif'])

```

#### For predicting Tc from a list of pymatgen structures [pmgs] 

``ms.data_remove_duplicates().structures[0:10]`` are the first 10 from the default structures saved.

```python
ms.predict_structure(modelms,ms.data_remove_duplicates().structures[0:10],structureFiles=False)

```

## Optimizing the structures

#### Optimizing for maximum Tc

```python
ms.superconductoropt(modelms,['C1S1H7.cif'])

```

#### Optimizing for picked Tc

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

```python
F=FourierSt(pr=True)
modeli,outputs=F.buildsample(modeltype='predictive',typea='original')
```

#### Build and fit prebuilt model build sample data

```python
F=FourierSt(pr=True)
outputs=F.sampledata(typea='original')
```

#### Build and fit prebuilt model or build sample data 

```python
modesl=F.samplemodels(modeltype='predictive',typea='original')
modesl=F.modelfromsample(modesl)
```
