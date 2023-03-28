
### Extra

Creating the models and data from beginning.

#### Make database structures 

```python
database=FourierSt(structures=aarrstuff3_primitive,filenames=[],compositionsfrac=ccdat.material,critical_temps=ccdat.critical_temp,pr=True)
custom = FourierSt(filenames=filenames,critical_temps=superconductcs,pr=True)
```

```python
FM_database=database.data_remove_duplicates().TforO()
FM_database=database.inputchar(typea='original')
FMucustom=custom.TforO()
FMucustom=custom.inputchar(typea='original')
```

```python
numrd_database=database.numrdata()
numrdcustom=custom.numrdata()
database_Tcs=database.set_avTc()
custom_Tcs=custom.critical_tempsdatapredictive=FourierSt(pr=True)
datapredictive.totalq=np.append(FM_database,FMucustom,axis=0)
datapredictive.totalz=np.append(numrd_database,numrdcustom,axis=0)
datapredictive.totalT=np.append(database_Tcs,custom_Tcs)
```

```python
model_predictive_s=datapredictive.samplemodels('predictive','original')
model_predictive_s=datapredictive.modelfromsample(model_predictive_s)
dfda,nmda=datapredictive.predict_structure(model_predictive_s,["C1S1H7.cif"],typea="original")
dfda,nmda=datapredictive.predict_structure(model_predictive_s,aarrstuff3_primitive[19:23],structureFiles=False,typea="original")
SC_O=datapredictive.superconductoropt(model_predictive_s,['C1S1H7.cif'],typea="original")
F_set=F.sampledata(typea="original")
sam=F.samplemodels('predictive','original')
model_i=F.modelfromsample(sam,epochs=300)
```
