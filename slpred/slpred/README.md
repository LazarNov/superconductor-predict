# model
<Use Predicting.ipynb to predict superconducting temperature of a file>

# Predict with saved models try ms=FourierSt() modeltype=``predictivesave'' for saved predictive model
ms=FourierSt()
modelms=ms.samplemodels(modeltype='predictivesave',typea='original')


# ms=FourierSt(pr=True) modeltype=''predictivesavepr'' for saved predictive model with more accuracte gaussians
ms=FourierSt(pr=True)
modelms=ms.samplemodels(modeltype='predictivesavepr',typea='original')


# For predicting Tc from a list [cifs] 
ms.predict_structure(modelms,['C1S1H7.cif'])


# For predicting Tc from a list of pymatgen structures [pmgs] 
ms.predict_structure(modelms,ms.data_remove_duplicates().structures[0:10],structureFiles=False)


# optimizing for maximum Tc
ms.superconductoropt(modelms,['C1S1H7.cif'])


# optimizing for picked Tc
ms.superconductoropt(modelms,['C1S1H7.cif'],optmax=False,target=180)


# Specify bounds and optimization latparamaters angles
ms.superconductoropt(modelms,['C1S1H7.cif'],lbounds=[[2,12],[2,12],[2,12],[60,120],[60,120],[60,120]],option={'method':'nelder-mead','tol':0.1,'options':{'maxfev':3000}})


# Show last optimization 
ms.last_opt


# Start optimization from last structure
minimaobj=ms.superconductoropt(modelms,[ms.last_opt],structuresfiles=False)
# make cif and crystal with Z= specifying atomic numbers 
makeAndWriteCrystal(minimaobj.x,fileName="optout.cif",buffer=[6,1,1,1,1,1,1,1,16])
# make cif from last optimization
pymatgen.io.cif.CifWriter(ms.last_opt['structure']).write_file('optout.cif')


# Generate and fit sample model and data 
F=FourierSt(pr=True)
modeli,outputs=F.buildsample(modeltype='predictive',typea='original')


# Build and fit prebuilt model build sample data 
F=FourierSt(pr=True)
outputs=F.sampledata(typea='original')


# Build and fit prebuilt model or build sample data 
modesl=F.samplemodels(modeltype='predictive',typea='original')
modesl=F.modelfromsample(modesl)

# make database structures 

database=FourierSt(structures=aarrstuff3_primitive,filenames=[],compositionsfrac=ccdat.material,critical_temps=ccdat.critical_temp,pr=True)
custom = FourierSt(filenames=filenames,critical_temps=superconductcs,pr=True)

FM_database=database.data_remove_duplicates().TforO()
FM_database=database.inputchar(typea='original')
FMucustom=custom.TforO()
FMucustom=custom.inputchar(typea='original')

numrd_database=database.numrdata()
numrdcustom=custom.numrdata()
database_Tcs=database.set_avTc()
custom_Tcs=custom.critical_tempsdatapredictive=FourierSt(pr=True)
datapredictive.totalq=np.append(FM_database,FMucustom,axis=0)
datapredictive.totalz=np.append(numrd_database,numrdcustom,axis=0)
datapredictive.totalT=np.append(database_Tcs,custom_Tcs)

model_predictive_s=datapredictive.samplemodels('predictive','original')
model_predictive_s=datapredictive.modelfromsample(model_predictive_s)
dfda,nmda=datapredictive.predict_structure(model_predictive_s,["C1S1H7.cif"],typea="original")
dfda,nmda=datapredictive.predict_structure(model_predictive_s,aarrstuff3_primitive[19:23],structureFiles=False,typea="original")
SC_O=datapredictive.superconductoropt(model_predictive_s,['C1S1H7.cif'],typea="original")
F_set=F.sampledata(typea="original")
sam=F.samplemodels('predictive','original')
model_i=F.modelfromsample(sam,epochs=300)