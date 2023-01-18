

import tensorflow as tf
import pymatgen,random,os,chemparse,shutil,scipy,cvnn,cmath,mendeleev,time,gc
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numbers import Integral
from functools import partial
from scipy.signal.signaltools import convolve2d
from numpy.lib.shape_base import kron
from scipy.ndimage import convolve

import tqdm
from tqdm import tqdm
import cvnn.layers as c_l

from numba import jit,vectorize, float64
from numpy import mean,std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score, RepeatedKFold,train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import accuracy_score


# Elements used, later should be expanded since superconductors with atoms above Rn are in dataset
elementslist = ["H","He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P",
  "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
  "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru",
  "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce",
  "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf",
  "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn"]



from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv3D, AveragePooling3D, MaxPooling3D, Dropout, BatchNormalization
from numpy.core.fromnumeric import shape
from keras.engine.input_layer import Input


Pathv=__path__[0]

# Some files for processing without running every time.
# Elemental contributions, compounds, and Tcs from supercon. 
ccdat = pd.read_csv(Pathv+'/unique_m1.csv') # Elemental contributions, compounds, and Tcs from supercon. 
# Table from Molecules2021,26, 8. https://dx.doi.org/10.3390/molecules26010008 as it was parsed.


aarrstuff3_primitive = np.load(Pathv+"/aarrstuff3_primitive.npy", allow_pickle=True) #MaterialsProject structures found from supercon data table earlier.
#allsctraindf = pd.read_csv("/content/drive/My Drive/train.csv") # Chemical composition-based data from Molecules2021,26, 8. https://dx.doi.org/10.3390/molecules26010008

# Cif file names of many computed predicted superconductors by 
# https://doi.org/10.1103/PhysRevB.104.054501
filenames=[
"500GPa_R-3m_SrH10.cif",
"500GPa_R-3m_CaH10.cif",
"500GPa_R-3_SrH24.cif",
"500GPa_Pm-3_MgH12.cif",
"500GPa_P-62m_SrH15.cif",
"500GPa_P-62m_CaH15.cif",
"500GPa_P6_3_mmc_NaH9.cif",
"500GPa_P3m1_MgH13.cif",
"500GPa_P-1_YH20.cif",
"500GPa_P-1_YH18.cif",
"500GPa_I4_mmm_Na2H11.cif",
"500GPa_Fm-3m_SrH10.cif",
"500GPa_Cmmm_Na2H11.cif",
"500GPa_C2_m_MgH10.cif",
"500GPa_C2_m_MgH8.cif",
"300GPa_R-3m_LiH6.cif",
"300GPa_P6_mmm_LiH2.cif",
"300GPa_P4_mmm_NaH5.cif",
"300GPa_P-1_ScH14.cif",
"300GPa_P-1_ScH12.cif",
"300GPa_Immm_ScH8.cif",
"300GPa_Im-3m_ScH6.cif",
"300GPa_Im-3m_MgH6.cif",
"300GPa_I4_mmm_MgH4.cif",
"300GPa_F-43m_YH9.cif",
"300GPa_Cmcm_LiH3.cif",
"300GPa_C2_m_NaH7.cif",
"300GPa_C2_m_LiH6.cif",
"200GPa_R-3m_Mg2H5.cif",
"200GPa_Pm-3m_NaH6.cif",
#"200GPa_P6_3mc_AcH12.cif",
"200GPa_P-1_MgH14.cif",
"200GPa_Im-3m_SH3.cif",
"200GPa_I4_mmm_NaH8.cif",
"200GPa_I4_mmm_MgH4.cif",
"200GPa_Fmmm_SH7.cif",
#"200GPa_Fmmm_AcH6.cif",
"200GPa_Fm-3m_MgH13.cif",
"200GPa_Cmmm_Na2H11.cif",
#"200GPa_Cmcm_AcH4.cif",
"200GPa_C2_m_Mg2H7.cif",
"200GPa_C2_m_LaH7.cif",
"100GPa_R-3_NaH24.cif",
"100GPa_Pm-3m_NaH6.cif",
"100GPa_P-1_LaH5.cif",
#"100GPa_P-1_AcH5.cif",
"100GPa_Im-3m_CaH6.cif",
"100GPa_Fmm2_NaH16.cif",
"100GPa_Cmmm_Na2H11.cif",
"100GPa_C2_m_KH10.cif",
#"100GPa_C2_m_AcH12.cif"
]

# Tcs for superconductors above
elirang=np.empty(shape=(1,2))
elirang=	np.append(elirang,[[190,	228]],axis=0)#	500
elirang=	np.append(elirang,[[184,	220]],axis=0)#	500
elirang=	np.append(elirang,[[218,	245]],axis=0)#	500
elirang=	np.append(elirang,[[360,	402]],axis=0)#	500
elirang=	np.append(elirang,[[110,	139]],axis=0)#	500
elirang=	np.append(elirang,[[134,	167]],axis=0)#	500
elirang=	np.append(elirang,[[235,	269]],axis=0)#	500
elirang=	np.append(elirang,[[257,	287]],axis=0)#	500
elirang=	np.append(elirang,[[212,	244]],axis=0)#	500
elirang=	np.append(elirang,[[213,	246]],axis=0)#	500
elirang=	np.append(elirang,[[76,	104]],axis=0)#	500
elirang=	np.append(elirang,[[285,	319]],axis=0)#	500
elirang=	np.append(elirang,[[141,	180]],axis=0)#	500
elirang=	np.append(elirang,[[232,	270]],axis=0)#	500
elirang=	np.append(elirang,[[91,	121]],axis=0)#	500

elirang=	np.append(elirang,[[130,	161]],axis=0)#	300
elirang=	np.append(elirang,[[177,	207]],axis=0)#	300
elirang=	np.append(elirang,[[138,	164]],axis=0)#	300
elirang=	np.append(elirang,[[91,	115]],axis=0)#	300
elirang=	np.append(elirang,[[137,	165]],axis=0)#	300
elirang=	np.append(elirang,[[212,	233]],axis=0)#	300
elirang=	np.append(elirang,[[135,	161]],axis=0)#	300
elirang=	np.append(elirang,[[271,	301]],axis=0)#	300
elirang=	np.append(elirang,[[59,	84]],axis=0)#	300
elirang=  np.append(elirang,[[261,	293]],axis=0)#	300
elirang=	np.append(elirang,[[112,	140]],axis=0)#	300
elirang=	np.append(elirang,[[167,	198]],axis=0)#	300
elirang=	np.append(elirang,[[130,	163]],axis=0)#	300

elirang=	np.append(elirang,[[24,	39]],axis=0)#	200
elirang=	np.append(elirang,[[260,	288]],axis=0)#	200
#elirang=	np.append(elirang,[[245,	280]],axis=0)#	200
elirang=	np.append(elirang,[[112,	134]],axis=0)#	200
elirang=  np.append(elirang,[[196,	219]],axis=0)#	200
elirang=	np.append(elirang,[[152,	175]],axis=0)#	200
elirang=	np.append(elirang,[[73,	94]],axis=0)#	200
elirang=  np.append(elirang,[[58,	78]],axis=0)#	200
#elirang=	np.append(elirang,[[169,	204]],axis=0)#	200
elirang=	np.append(elirang,[[196,	224]],axis=0)#	200
elirang=	np.append(elirang,[[129,	162]],axis=0)#	200
#elirang=	np.append(elirang,[[42,	58]],axis=0)#	200
elirang=	np.append(elirang,[[56,	75]],axis=0)#	200
elirang=	np.append(elirang,[[105,	134]],axis=0)#	200

elirang=	np.append(elirang,[[40,	55]],axis=0)#	100
elirang=	np.append(elirang,[[248,	279]],axis=0)#	100
elirang=	np.append(elirang,[[40,	58]],axis=0)#	100
#elirang=	np.append(elirang,[[49,	69]],axis=0)#	100
elirang=	np.append(elirang,[[216,	253]],axis=0)#	100
elirang=	np.append(elirang,[[61,	75]],axis=0)#	100
elirang=	np.append(elirang,[[134,	161]],axis=0)#	100
elirang=	np.append(elirang,[[134,	157]],axis=0)#	100
#elirang=	np.append(elirang,[[13,	20]],axis=0)#	100
elirang=elirang[1:]
superconductcs=[np.mean(eli_) for eli_ in elirang]

#function that changes a cif file to pymatgen format. composition based feats 

def StoA(fileName):
  struc_di={} 
  str_dic={}
  structure=pymatgen.core.structure.IStructure.from_file(filename=fileName,primitive=True)
  print(structure)
  str_dic['structure']=structure
  comp1 = structure.composition
  ncomp3 = comp1.reduced_formula
  feat=np.array(Extract(ncomp3,ss_e_d))
  for i in range(len(elementslist)):
    feat = np.append(feat,comp1[i+1])
  struc_di[ncomp3]=feat
  return str_dic, struc_di

#
# These are a set of functs that augment the data from supercon. If there isn't any update to the data, some will not need calling
# # 


#Gets supercon format of chemical composition to MaterialsProject format for data search.


#This checks the number of digits needed the composition in supercon list to make whole number compositions for MaterialsProject
def chemNorm(cform):
  fracCheck = False
  for key in cform.keys():
    #cform[key] = round(cform[key], 6) #float("{:.6f}".format(cform[key]))
    if cform[key].is_integer() == False:
      fracCheck = True
   
  while(fracCheck):
    fracCheck = False
    for key in cform.keys():
      cform[key] = 10*cform[key]
      cform[key] = round(cform[key], 6) #float("{:.6f}".format(cform[key]))
      if  cform[key].is_integer() == False: #<1.0:
        fracCheck = True

  return cform



# # readablematerials: dataframe of the material names and Tcs T(y), y=composition, reported on the database.
# # returns: Dictionaries with mean(T(y)), variance(T(y)), T(y).
# def supercon_average_Tc(readablematerials,debugcheck=False):
#   import statistics
#   dupmats = []
#   global averagedTcs
#   crtclVECTOR = {}
#   TcsVector = {}
#   averagedTcs ={}
#   varianceTc = {}
#   filledarray = []
#   # Get the list of all Tcs for each compound from supercon.

#   for compnames1 in readablematerials["material"]:
#     if compnames1 in filledarray: #tcbynameavg:
#       dupmats.append(compnames1)
#     varianceTc[compnames1]=0
#     filledarray.append(compnames1)

#   for dups in range(0,len(dupmats)): 
#     avgmat = []
#     hits = (i for i,value in enumerate(readablematerials["material"]) if value == dupmats[dups])

#     for hit in hits:
#       avgmat.append(readablematerials["critical_temp"][hit])
#     avgoflist = statistics.mean(avgmat)
#   # The T(y), variance(T(y)), mean(T(y))
#     crtclVECTOR[dupmats[dups]]=avgmat
#     varianceTc[dupmats[dups]] = statistics.variance(avgmat)
#     averagedTcs[dupmats[dups]] = avgoflist
  
#   return averagedTcs, varianceTc, crtclVECTOR


#   # iterating through the elements of list 
#   for colnam in elementcolumnnames: 
#       posattmatf[colnam] = -12000 #-1
#   print(posattmatf) #posattmat before
#   return posattmatf


def numrdata(datamat,numdit): # posattmat
  
  featdit=[]
  for ind in range(0,len(datamat)):
    feats=[]

    comp1 = datamat[ind]['structure'].composition
    ncomp3 = comp1.reduced_formula
    print(ncomp3)
    # matcomp3 = chemparse

    feats.append(datamat[ind]['structure'].lattice.a)
    feats.append(datamat[ind]['structure'].lattice.b)
    feats.append(datamat[ind]['structure'].lattice.c)
    feats.append(datamat[ind]['structure'].lattice.angles[0])
    feats.append(datamat[ind]['structure'].lattice.angles[1])
    feats.append(datamat[ind]['structure'].lattice.angles[2])
    feats.append(datamat[ind]['structure'].lattice.volume)
    feats.append(pymatgen.core.Structure.get_space_group_info(datamat[ind]['structure'])[1])
    

    feats=np.append(np.array(feats),np.array(numdit[ncomp3]))
    
    featdit.append(feats)
  
  return featdit
def bestTcs(datamat,allAvgTc,crtklv):
  tcs1 = []
  StatTc={}
  weights={}
  for crkt in crtklv:
    StatTc[crkt]=np.array([])
    weights[crkt]=[[],[]]
  for ind in range(0,len(datamat)):
    comp1 = datamat[ind]['structure'].composition
    ncomp3 = comp1.reduced_formula
    #print(ncomp3)

    if ncomp3 in list(crtklv.keys()):
      Tcchoice=random.choice(crtklv[ncomp3])
      tcs1.append(Tcchoice)

      StatTc[ncomp3]=np.append(StatTc[ncomp3],Tcchoice)
      weights[ncomp3][0].append(83)
      weights[ncomp3][1].append(Tcchoice)
    else:
      tcs1.append(allAvgTc[ncomp3])
  return tcs1,StatTc,weights


def bestTcsb(datamat,allAvgTc,crtklv,weights):
  tcs1 = []
  StatTc={}
  for crkt in crtklv:
    StatTc[crkt]=np.array([])
  for ind in range(0,len(datamat)):
    comp1 = datamat[ind]['structure'].composition
    ncomp3 = comp1.reduced_formula

    if ncomp3 in list(crtklv.keys()) :
      if weights[ncomp3][0][len(StatTc[ncomp3])]>random.randrange(0,100):
        poolint=np.append(crtklv[ncomp3],np.mean(crtklv[ncomp3]))
        Tcchoice=random.choice(poolint)
        print(ncomp3,len(StatTc[ncomp3])," mutated ",weights[ncomp3][0][len(StatTc[ncomp3])],"% rate from ", weights[ncomp3][1][len(StatTc[ncomp3])]," to ",Tcchoice, " pool: ",poolint)
        weights[ncomp3][0][len(StatTc[ncomp3])]=21
        weights[ncomp3][1][len(StatTc[ncomp3])]=Tcchoice
        StatTc[ncomp3]=np.append(StatTc[ncomp3],Tcchoice)
        tcs1.append(Tcchoice)
      else: 
        poolint=np.append(crtklv[ncomp3],np.mean(crtklv[ncomp3]))
        print(ncomp3,len(StatTc[ncomp3])," remained ", weights[ncomp3][0][len(StatTc[ncomp3])],"% rate ", weights[ncomp3][1][len(StatTc[ncomp3])]," pool:",poolint)
        tcs1.append(weights[ncomp3][1][len(StatTc[ncomp3])])
        StatTc[ncomp3]=np.append(StatTc[ncomp3],weights[ncomp3][1][len(StatTc[ncomp3])])
    else:
      tcs1.append(allAvgTc[ncomp3])
  return tcs1,StatTc,weights



# Code that calculates some composition feats   Molecules2021,26, 8. https://dx.doi.org/10.3390/molecules26010008
# The formulas are in the manuscript.# code that calculates the composition feats.
import rpy2.robjects
from rpy2.robjects import packages
from rpy2.robjects import pandas2ri


pandas2ri.activate()
#utils = packages.importr('utils')
#base =  packages.importr('base')
#utils.chooseCRANmirror()
# utils.install_packages("CHNOSZ")
# utils.install_packages("xgboost")
CHN=packages.importr("CHNOSZ")
xgb=packages.importr("xgboost")

rpy2.robjects.r.load(Pathv+"/tc.RData") #tc.RData is from Molecules2021,26, 8. https://dx.doi.org/10.3390/molecules26010008,https://github.com/khamidieh/predict_tc
ss_e_d=rpy2.robjects.r['subset_element_data']
Extract=rpy2.robjects.r['extract'] # This function takes composition and evaluates the characterstics.

import keras_tuner as kt
from tensorflow import keras
from keras.layers import Activation
from keras.models import Model
# Creating a sample model for the compositional part of data
def createmlmaterial(d,inputshape):
  
  model=Dense(256,input_dim=inputshape,activation="relu")(d)
  model = BatchNormalization(center=True, scale=True)(model)
  #model=Dropout(0.5)(model)
  model=Dense(256,activation="relu")(model)
  model = BatchNormalization(center=True, scale=True)(model)
  #model=Dropout(0.5)(model)
  model=Dense(256,activation="relu")(model)
  model = BatchNormalization(center=True, scale=True)(model)
  #model=Dropout(0.5)(model)
  model=Dense(256,activation="relu")(model)
  model = BatchNormalization(center=True, scale=True)(model)
  #model=Dropout(0.5)(model)
  # model=Dense(512,activation="relu")(model)
  # model = BatchNormalization(center=True, scale=True)(model)
  # model=Dense(512,activation="relu")(model)
  # model = BatchNormalization(center=True, scale=True)(model)
  return model
# Creating a sample model for the density part.
def createconvolutionlayers(inputs,inputshape):
  keras.backend.clear_session()
  gc.collect()
  # x = c_l.ComplexConv3D(32, kernel_size=(3 ,3, 3), activation='cart_relu',input_shape=inputshape)#
  # x = (c_l.ComplexAvgPooling3D(pool_size=(2, 2, 2)))(x)
  # x = c_l.ComplexBatchNormalization(center=True, scale=True)(x)
  # x=c_l.ComplexDropout(0.50)(x)
  # x = c_l.ComplexFlatten()(x)
  # x = c_l.ComplexDense(units=540,activation='convert_to_real_with_abs')(x)
  x = inputs
  x = c_l.ComplexConv3D(32, kernel_size=(3), activation='cart_relu')(x)
  x = c_l.ComplexAvgPooling3D(pool_size=(2))(x)
  x=c_l.ComplexDropout(0.50)(x)
  
  x = c_l.ComplexConv3D(32, kernel_size=(3), activation='convert_to_real_with_abs')(x)
  x = c_l.ComplexAvgPooling3D(pool_size=(2))(x)
  # x = c_l.ComplexBatchNormalization(center=True, scale=True)(x)
  x=c_l.ComplexDropout(0.50)(x)
  x = c_l.ComplexFlatten()(x)
  #x=concatenate([c_l.ComplexFlatten()(inputs),x])
  #x = c_l.ComplexDense(units=128,activation='zrelu')(x)
  #x = c_l.ComplexDense(units=128,activation='zrelu')(x)
  # x = c_l.ComplexDense(units=128,activation='zrelu')(x)
  # x = c_l.ComplexDense(units=128,activation='zrelu')(x)

  #x = c_l.ComplexDense(units=256,activation='convert_to_real_with_abs')(x)

  return x#model
  
from keras.layers import concatenate
# concatinating together 
def concatmodel(inputsA,inputsB,input_shapeA,epochs=100):
  keras.backend.clear_session()
  gc.collect()
  modela=createconvolutionlayers(inputsA,input_shapeA)

  modeln=createmlmaterial(inputsB,175)
  models=concatenate([modela,modeln])
  models=Dense(512,activation="relu")(models)
  models = BatchNormalization(center=True, scale=True)(models)
  #models = Dropout(0.50)(models)
  models=Dense(512,activation="relu")(models)
  models= BatchNormalization(center=True, scale=True)(models)
  #models = Dropout(0.50)(models)  
  models=Dense(512,activation="relu")(models)
  models= BatchNormalization(center=True, scale=True)(models)
  models=Dense(512,activation="relu")(models)
  models= BatchNormalization(center=True, scale=True)(models)
  outs = Dense(1,activation='relu')(models)
  models = Model(inputs=[inputsA,inputsB],outputs=[outs])
  return models

Z_atomic = {}
indexS =1
for elment in elementslist:
  Z_atomic[elment]=indexS
  indexS = indexS + 1
  # Atomic mass for weights N_Z m(Z)
Z_mass={}
for element in elementslist:
  Z_mass[element]=mendeleev.element(element).mass
      
  # Atomic radii sigma_Z r_Z
Z_r = {}
for element in elementslist:
  Z_r[element]=pymatgen.core.periodic_table.Element(element).atomic_radius
from keras.callbacks import ModelCheckpoint
# for saving low val loss
checkpoint = ModelCheckpoint('modelfile',initial_value_threshold=50., monitor='val_loss', verbose=1, \
                             save_best_only=True, save_weights_only=True, \
                             mode='auto', save_frequency=1)

def cifpred(filename): #ciffile
  s10,adit=StoA(filename)
  numdit[list(adit.keys())[0]]=adit[list(adit.keys())[0]]
  csH=Tfor([s10,s10],delta=13,gamvar=[0.25,0.0025],syref=False) # processing match model
  csHn=numrdata([s10,s10],numdit)
  cshM=inputchar(csH,"original")
  Tcs=models.predict([cshM[1:],np.array(csHn)[1:]])
  print("T:",Tcs)
  return csHn,cshM,s10,Tcs #numda

def mixture(y_true,y_pred):     
        return tf.keras.losses.huber(y_true,y_pred,delta=7)

from keras import backend as kb
def coeff_determination(y_true, y_pred): # coeffdet function use
    return ( 1 - kb.sum(kb.square( y_true-y_pred ))/(kb.sum(kb.square( y_true - kb.mean(y_true) ) ) + kb.epsilon()) )

  
# sigmaS = sigma_Z
# strct_hkl = h, k, or l
# latm = a, b, c
# position = position in fractional coordinates
# positionq = position in cartesian coordinates
# structure = structure object
#  Z = Factors used for weights for atoms N_Z
#  r0 = Factors used for sigma_Z, only needed for charge property
# Factor in Fourier series
def f(sigmaS,strct_hkl,latm):
  return cmath.exp(-2 * cmath.pi**2 *sigmaS*strct_hkl**2 /(latm**2))

# This calcuates the phase from the position

def phi(strct_hkl,position,latm):
  return cmath.exp(2 *cmath.pi * cmath.sqrt(-1) * position *strct_hkl) #/ latm

# This is a combination of the two terms for calculation purposes
# positions in fractional coordinates.
def Fphi(strct_hkl,positionq,latm,sigmaS):
  return cmath.exp(2 *cmath.pi * (-1 * cmath.pi* sigmaS*strct_hkl**2  +  cmath.sqrt(-1) * positionq*latm *strct_hkl)) #/ latm
# Just exponential term computational simplicty. Fractional
@jit
def SFphi(strct_hkl,position,latm,sigmaS):
  
  return (2 *cmath.pi * (-1 * cmath.pi* sigmaS*strct_hkl**2  +  cmath.sqrt(-1) * position*latm *strct_hkl)) #/ latm
@jit
def SFphi0(strct_hkl,position,latm,sigmaS):
  
  return (2 *cmath.pi * (-1 * cmath.pi* sigmaS*strct_hkl**2  +  cmath.sqrt(-1) * position*latm *strct_hkl)) #/ latm
# Just exponential term.
@jit
def SFphiA(strct_hkl,position,latm,sigmaS):
  #position in fractional.
  return (2 *cmath.pi * (-1 * cmath.pi* sigmaS*(strct_hkl/latm)**2  -  cmath.sqrt(-1) * position *strct_hkl)) #/ latm
#This calculates the fourier factor of the structure of all atoms.
@jit
def calculatingS(positionV,lat,sigma,z,r,atomd,h,k,l):
  structureconst=0.0
  for atoms in range(len(z)):
    Z_at=z[atoms]
    if atomd:
      r_at=r[atoms]
      s_tD = sigma*r_at
    else:
      s_tD= sigma*Z_at
    structureconst += Z_at *cmath.exp(SFphiA(h,positionV[atoms][0],lat[0],s_tD)+SFphiA(k,positionV[atoms][1],lat[1],s_tD)+SFphiA(l,positionV[atoms][2],lat[2],s_tD))
  structureconst=structureconst/lat[0]/lat[1]/lat[2]
  return structureconst
@jit
def calculatingSa(positionV,lat,vol,sigma,z,r,atomd,h,k,l):
  structureconst=0.0
  for atoms in range(len(z)):
    Z_at=z[atoms]
    if atomd:
      r_at=r[atoms]
      s_tD = sigma*r_at
    else:
      s_tD= sigma*Z_at
    structureconst += Z_at *cmath.exp(SFphi(h,positionV[atoms][0],lat[0],s_tD)+SFphi(k,positionV[atoms][1],lat[1],s_tD)+SFphi(l,positionV[atoms][2],lat[2],s_tD))
  structureconst=structureconst/vol
  return structureconst
# Calculates Fourier values on the domain by input delta/syref
# vecP = matrix of positions for the structure
# zvec = vector of the weights for N_Z
# latvec = vector of a,b,c
# r = vector of sigma_Z weights
def F_T_o(structure,Z,r0=0):
  positions_vector = np.zeros((len(structure['structure'].sites),3))
  z=np.zeros(len(structure['structure']))
  r=np.zeros(len(structure['structure']))
  lats=np.zeros(3)
  for indlat in range(3):
    lats[indlat]=structure['structure'].lattice.abc[indlat]

  for index_sites in range(len(structure['structure'].sites)):
    z[index_sites]=Z[str(structure['structure'].species[index_sites])]
    if r0 != 0:
      r[index_sites]=r0[str(structure['structure'].species[index_sites])]

    for ind in range(3):
      positions_vector[index_sites,ind]= structure['structure'].sites[index_sites]._frac_coords[ind]
  return positions_vector,lats,z,r
def F_T_o_s(structure,Z,r0=0):
  positions_vector = np.zeros((len(structure['structure'].sites),3))
  z=np.zeros(len(structure['structure']))
  r=np.zeros(len(structure['structure']))
  #lats=np.zeros(3)
  
  lats=inverses(structure['structure'].lattice.matrix )
  vol=structure['structure'].lattice.volume
  for index_sites in range(len(structure['structure'].sites)):
    z[index_sites]=Z[str(structure['structure'].species[index_sites])]
    if r0 != 0:
      r[index_sites]=r0[str(structure['structure'].species[index_sites])]

    for ind in range(3):
      positions_vector[index_sites,ind]= structure['structure'].sites[index_sites].coords[ind]
  return positions_vector,lats,z,r,vol

@jit
def inverses(latt):

  E_=np.linalg.inv(latt)

  return E_

def FsmoothSTre(structure,Z,delta,atomd,var,syref=False,r0=0):
  if syref:
    domaine=np.arange(-(delta-1)/2,(delta+1)/2,1)
  else:
    domaine=np.arange(0,delta,1)

  Fmat = np.zeros( [1, delta,delta, delta ],dtype=np.complex64)
  vecP,latvec,zvec, r=F_T_o(structure, Z, r0)
  Hind=-1

  Fmat= fLoop(Fmat ,vecP,zvec,latvec,domaine,r,atomd,var)
  return Fmat  
def Fsmooth(structure,Z,delta,atomd,var,syref=False,r0=0):
  if syref:
    domaine=np.arange(-(delta-1)/2,(delta+1)/2,1)
  else:
    domaine=np.arange(0,delta,1)

  Fmat = np.zeros( [1, delta,delta, delta ],dtype=np.complex64)
  vecP,latvec,zvec, r,vol=F_T_o_s(structure, Z, r0)
  Hind=-1

  Fmat= fLoopp_(Fmat ,vecP,vol,zvec,latvec,domaine,r,atomd,var)
  return Fmat  
@jit
def fLoop(Fmat ,vecP,zvec,latvec,domaine,r=0, atomd=False, var = 0.0025 ):
  Hind=-1

  for pointH in domaine:
    Hind=Hind+1
    Kind=-1
    for pointK in domaine:
      Kind=Kind+1
      Lind=0
      for pointL in domaine:

        Fmat[0,Hind,Kind,Lind] = calculatingS(vecP,latvec,var,zvec,r,atomd,pointH,pointK,pointL)

        Lind = Lind +1    

  return Fmat

def fLoopp_(Fmat ,vecP,vol,zvec,latvec,domaine,r=0, atomd=False, var = 0.0025 ):
  Hind=-1

  for pointH in domaine:
    Hind=Hind+1
    Kind=-1
    for pointK in domaine:
      Kind=Kind+1
      Lind=0
      for pointL in domaine:
        mult=np.linalg.multi_dot((latvec,[pointH,pointK,pointL]))
        Fmat[0,Hind,Kind,Lind] = calculatingSa(vecP,np.array([1,1,1]),vol,var,zvec,r,atomd,mult[0],mult[1],mult[2])

        Lind = Lind +1    

  return Fmat

#returns a pymatgen Lattice object corresponding to the found minima's lattice parameters
def makeLattice(minimaObj):
  crystal = pymatgen.core.lattice.Lattice.from_parameters(a=minimaObj[0], b = minimaObj[1], c = minimaObj[2],alpha = minimaObj[3], beta = minimaObj[4], gamma = minimaObj[5])
  return crystal
#populates the minima's corresponding Lattice object with its corresponding a atomic species and positions
def makeCrystal(lat, minimaObj,buffer):
  crystalStruct = pymatgen.core.structure.IStructure(lattice = lat, species = buffer, coords = np.reshape(minimaObj[6:],(int(len(buffer)),3)), to_unit_cell = True,)
  return(crystalStruct)
#uses the above two functions to generate a cif file with the found minima's corresponding, lattice, molecular species and positions, with a given file name
#file name defaults to "crystalStruct.cif"
import pymatgen.io.cif

def makeAndWriteCrystal(minimaObj, fileName = "crystalStruct.cif",buffer=0,write_f=True):
  crystalStruct = makeCrystal(makeLattice(minimaObj), minimaObj,buffer)
  if write_f:
    pymatgen.io.cif.CifWriter(crystalStruct).write_file(filename=fileName)
  return crystalStruct 

class dataframe:# Code that calculates some composition feats   Molecules2021,26, 8. https://dx.doi.org/10.3390/molecules26010008
# The formulas are in the manuscript.
  def __init__(self,compositionsfrac=ccdat.material,critical_temps=ccdat.critical_temp):
    
    self.compositionsfrac=compositionsfrac
    self.critical_temps=critical_temps

 
  def renamecomp(self):
    import pymatgen as pmg
    def chemNorm(cform):
      fracCheck = False
      for key in cform.keys():
       
        if cform[key].is_integer() == False:
          fracCheck = True
      
      while(fracCheck):
        fracCheck = False
        for key in cform.keys():
          cform[key] = 100*cform[key]
          cform[key] = round(cform[key], 6) 
          if  cform[key].is_integer() == False: #<1.0:
            fracCheck = True

      return cform      

    ccdatfunc= dataframe()
    materials=[]
    for M in self.compositionsfrac:

      normcf2=chemNorm(chemparse.parse_formula(M))
      ncomp2=pmg.core.composition.Composition(normcf2)
      mplook2= ncomp2.reduced_formula
      materials.append(mplook2)
    materials=np.array(materials)
    ccdatfunc=dataframe(materials,self.critical_temps)
    self.compositionsfrac=materials
    return ccdatfunc
  

  def set_Tc_arrays(self):
    Tcsdict={}
    for mats in self.compositionsfrac:
      Tcsdict[mats]=[]
    return Tcsdict

  def set_mat_names(self):
    setmatnames={}
    for mats in self.compositionsfrac:
      setmatnames[mats]=0
    return list(setmatnames.keys())
  def get_Tc_ve(self):
    #mpframe=self.renamecomp()
    Tcsdict=self.set_Tc_arrays()
    for indmats in range(len(self.compositionsfrac)):
      Tcsdict[self.compositionsfrac[indmats]].append(self.critical_temps[indmats])
    return Tcsdict
  def get_Tcmean(self):
    Tcs=self.get_Tc_ve()
    Tcmean={}
    
    for Tclist in Tcs.keys():
      Tcmean[Tclist]=mean(Tcs[Tclist])
    return Tcmean
  def get_varTc(self):
    Tcs=self.get_Tc_ve()
    varTcs={}
    for Tclist in Tcs.keys():
      varTcs[Tclist]=np.var(Tcs[Tclist])
    return varTcs
  def varTcs(self,cut):
    Tcs=self.get_Tc_ve()
    varTcbelow={}
    varTcabove={}
    varTcs=self.get_varTc()
    for Tc_S in tqdm(Tcs.keys()):
      if cut>=varTcs[Tc_S]:
        varTcbelow[Tc_S]=Tcs[Tc_S]
      else:
        varTcabove[Tc_S]=Tcs[Tc_S]
    return varTcbelow,varTcabove


class structuresinput(dataframe):
  

  def __init__(self,structures=aarrstuff3_primitive,filenames=[],compositionsfrac=None,critical_temps=None):
    self.filenames=filenames
    self.compositionsfrac=compositionsfrac  # Atomic numbers used for weights N_Z e(Z)
    if filenames==[]:

      self.structures=structures
    else:
      Structures_=[]
      for strfile in filenames:
        str_dic={}
        structure = pymatgen.core.structure.IStructure.from_file(filename=strfile,primitive=True)
        str_dic['structure']=structure
        Structures_.append(str_dic)
      self.structures=Structures_
      self.compositionsfrac=self.get_material_compositions()
    self.critical_temps=critical_temps
    dataframe.__init__(self,compositionsfrac=self.compositionsfrac,critical_temps=critical_temps)
    

  # strctureinput = structure obj
  # delta = max value for h, k, l
  # atomd = True if using charge distribution
  # var = variance coefficient gamma
  # syref = True if calculating negative term
  # Fmat =  matrix with Fourier values
  def data_remove_duplicates(self):
    def check_for_dup(infoarray):
      norepeatmaterials = []
      extraposition = []
      for matindexs in range(len(infoarray)):
        if infoarray[matindexs]['material_id'] in norepeatmaterials:
          extraposition.append(matindexs)
          print(matindexs,infoarray[matindexs]['material_id'])
        else:     
          norepeatmaterials.append(infoarray[matindexs]['material_id'])
      return extraposition
    norepeatdata=[]
    norepeatmaterialids=[]
    for superconductors in self.structures:
      if superconductors['material_id'] not in norepeatmaterialids:
        norepeatdata.append(superconductors)
        norepeatmaterialids.append(superconductors['material_id'])
    self.structures=norepeatdata
    return self
  def numditd(self):
    #def numdit_a(structure):
    struc_di={}
    for structure in self.structures:
      comp1 = structure['structure'].composition
      ncomp3 = comp1.reduced_formula
      feat=np.array(Extract(ncomp3,ss_e_d))

      for i in range(len(elementslist)):
        feat = np.append(feat,comp1[i+1])
      struc_di[ncomp3]=feat
    return struc_di



  def numrdata(self): # posattmat
    
    numdit=self.numditd()
    featdit=[]
    for ind in tqdm(range(len(self.structures))):
      feats=[]

      comp1 = self.structures[ind]['structure'].composition
      ncomp3 = comp1.reduced_formula
      #print(ncomp3)
      # matcomp3 = chemparse

      feats.append(self.structures[ind]['structure'].lattice.a)
      feats.append(self.structures[ind]['structure'].lattice.b)
      feats.append(self.structures[ind]['structure'].lattice.c)
      feats.append(self.structures[ind]['structure'].lattice.angles[0])
      feats.append(self.structures[ind]['structure'].lattice.angles[1])
      feats.append(self.structures[ind]['structure'].lattice.angles[2])
      feats.append(self.structures[ind]['structure'].lattice.volume)
      feats.append(pymatgen.core.Structure.get_space_group_info(self.structures[ind]['structure'])[1])
      

      feats=np.append(np.array(feats),np.array(numdit[ncomp3]))
      
      featdit.append(feats)
    
    return featdit

  def filetoS(self):
    structures=[]
    for strfile in self.filenames:
      str_dic={}
      structure = pymatgen.core.structure.IStructure.from_file(filename=strfile,primitive=True)
      str_dic['structure']=structure
      structures.append(str_dic)
    return structuresinput(structures=structures,filenames=self.filenames)
  def get_material_compositions(self):
      materialcompositions=[]
      # if files:
      #   structures=self.filetoS().structures
      # else:
      #   structures=self.structures
      for structure in self.structures:

        comp1 = structure['structure'].composition
        ncomp3 = comp1.reduced_formula
        materialcompositions.append(ncomp3)
      return materialcompositions
  def set_avTc(self):
    ''"Do not use if you have filenames that are polymorphisms"''
    self.renamecomp()
    means=self.get_Tcmean()
    Tcs=[]
    for compositions in  self.get_material_compositions():
      Tcs.append(means[compositions])
    return Tcs
class FourierSt(structuresinput):
  # Takes structures and weights to generate Fourier terms given input parameters.
  # Returns for charge and mass density.
  def __init__(self,delta=13,Z_atomic=Z_atomic,Z_mass=Z_mass,r0=Z_r,gamvar=[1./4,0.0025],syref=False,structures=aarrstuff3_primitive,filenames=[],compositionsfrac=None,critical_temps=None,pr=False):
    self.delta=delta
    self.Z_atomic=Z_atomic
    self.Z_mass=Z_mass
    self.r0=r0
    self.gamvar=gamvar
    self.syref=syref
    self.pr=pr
    structuresinput.__init__(self,structures=structures,filenames=filenames,compositionsfrac=compositionsfrac,critical_temps=critical_temps)

  from tqdm import tqdm
  def Tfor(self):
    non_iT=0
    for structure_materials in tqdm(self.structures):
      if non_iT==0:
        Fb=FsmoothSTre(structure_materials,self.Z_atomic,self.delta,atomd=True,syref=self.syref, var=self.gamvar[0],r0=self.r0)
        Fbp=FsmoothSTre(structure_materials,self.Z_mass,self.delta,atomd=False,syref=self.syref, var=self.gamvar[1])
      else:  
        Fb=np.append(Fb,FsmoothSTre(structure_materials,self.Z_atomic,self.delta,atomd=True,syref=self.syref,var=self.gamvar[0],r0=self.r0),axis=0)
        Fbp=np.append(Fbp,FsmoothSTre(structure_materials,self.Z_mass,self.delta,atomd=False,syref=self.syref,var=self.gamvar[1]),axis=0)
      #print(non_iT)
      non_iT +=1
    self.F= [Fb ,Fbp]
    return [Fb, Fbp]
  def TforO(self):
    non_iT=0
    for structure_materials in tqdm(self.structures):
      if non_iT==0:
        Fb=Fsmooth(structure_materials,self.Z_atomic,self.delta,atomd=True,syref=self.syref, var=self.gamvar[0],r0=self.r0)
        Fbp=Fsmooth(structure_materials,self.Z_mass,self.delta,atomd=False,syref=self.syref, var=self.gamvar[1])
      else:  
        Fb=np.append(Fb,Fsmooth(structure_materials,self.Z_atomic,self.delta,atomd=True,syref=self.syref,var=self.gamvar[0],r0=self.r0),axis=0)
        Fbp=np.append(Fbp,Fsmooth(structure_materials,self.Z_mass,self.delta,atomd=False,syref=self.syref,var=self.gamvar[1]),axis=0)
      #print(non_iT)
      non_iT +=1
    self.F= [Fb ,Fbp]
    return [Fb, Fbp]
  # Converts Fourier values, complex, to another form.
  def inputchar(self,typea):
    Fb,Fbp = self.F[0],self.F[1]
    if typea=="abs":
      Fm=np.array([abs(Fb),abs(Fbp)])
    if typea=="real":
      Fm=np.array([np.real(Fb),np.real(Fbp)])
    if typea=="complex":
      Fm=np.array([np.real(Fb),np.imag(Fb),np.real(Fbp),np.imag(Fbp)])
    if typea=="imag":
      Fm=np.array([np.imag(Fb),np.imag(Fbp)])
    if typea=="original":
      Fm=np.array([Fb,Fbp])

    Fm=Fm.transpose(1,2,3,4,0)
    return Fm

    #Converts pymatgen structure
    #to positions lattice parms and weights for N_Z and for sigma_Z if radius used

  def samplemodels(self,modeltype,typea,):
    
    if typea == "complex":
      fltrs= 4
    else:
      fltrs=2

    delta=self.delta
    input_shapeA=(delta, delta, delta,fltrs)
    # if model
    # inputsA=keras.Input(shape=input_shapeA)assdfasdsdfaa
    # else:asdfasdfasdfasdfasdfasdfadsfad
    inputsA = c_l.complex_input(shape=input_shapeA)
    inputsB = keras.Input(shape=(175))
    if modeltype=="predictive":
      models=concatmodel(inputsA,inputsB,input_shapeA)
      # if:
      # if
    elif modeltype=="predictivesave":
      models=tf.keras.models.load_model(Pathv+'/predictive',custom_objects={'mixture':mixture,'coeff_determination':coeff_determination})
    elif modeltype=="predictivesavepr":
      models=tf.keras.models.load_model(Pathv+'/predictivesavepr',custom_objects={'mixture':mixture,'coeff_determination':coeff_determination})
    elif modeltype=="r2model":
      models=tf.keras.models.load_model(Pathv+'/complexmodel',custom_objects={'mixture':mixture,'coeff_determination':coeff_determination})#concatmodelr2rd(inputsA,inputsB,input_shapeA)
    else:
      print('choose:``predictive``, ``predictivesave``, ``predictiveprsave``, ``r2model`` ')
      return
    return models
  def sampledata(self,typea="original",seed=42,split=0.80):
    database=FourierSt(self.delta,self.Z_atomic,self.Z_mass,self.r0,self.gamvar,self.syref,structures=aarrstuff3_primitive,filenames=[],compositionsfrac=ccdat.material,critical_temps=ccdat.critical_temp,pr=self.pr)
    custom = FourierSt(self.delta,self.Z_atomic,self.Z_mass,self.r0,self.gamvar,self.syref,filenames=filenames,critical_temps=superconductcs,pr=self.pr)
    if self.pr:

      FM_database=database.data_remove_duplicates().TforO()
      FM_database=database.inputchar(typea)
      FMucustom=custom.TforO()
      FMucustom=custom.inputchar(typea)
    else:
      FM_database=database.data_remove_duplicates().Tfor()
      FM_database=database.inputchar(typea)
      FMucustom=custom.Tfor()
      FMucustom=custom.inputchar(typea)

    numrd_database=database.numrdata()
    numrdcustom=custom.numrdata()
    database_Tcs=database.set_avTc()
    custom_Tcs=custom.critical_temps
    totalq=np.append(FM_database,FMucustom,axis=0)
    totalz=np.append(numrd_database,numrdcustom,axis=0)
    totalT=np.append(database_Tcs,custom_Tcs)
    descriptor_trainA,descriptor_testA,descriptor_trainb,descriptor_testb, Tc_train,Tc_test = train_test_split(totalq,totalz,totalT,random_state=seed, train_size=split)
    self.totalq=totalq
    self.totalz=totalz
    self.totalT=totalT
    self.descriptor_trainA=descriptor_trainA
    self.descriptor_testA=descriptor_testA
    self.descriptor_trainb=descriptor_trainb
    self.descriptor_testb=descriptor_testb
    self.Tc_train=Tc_train
    self.Tc_test=Tc_test
    
    return optimalTarget(database=database,custom=custom,FM_database=FM_database,FMucustom=FMucustom,numrd_database=numrd_database,numrdcustom=numrdcustom,database_Tcs=database_Tcs,custom_Tcs=custom_Tcs)
  def buildsample(self,modeltype="predictivesavepr",typea="original",seed=42,func="mixture",split=0.80,epochs=300):
    models=self.samplemodels(modeltype,typea)
    database=FourierSt(self.delta,self.Z_atomic,self.Z_mass,self.r0,self.gamvar,self.syref,structures=aarrstuff3_primitive,filenames=[],compositionsfrac=ccdat.material,critical_temps=ccdat.critical_temp,pr=self.pr)
    custom = FourierSt(self.delta,self.Z_atomic,self.Z_mass,self.r0,self.gamvar,self.syref,filenames=filenames,critical_temps=superconductcs,pr=self.pr)
    
    if self.pr:

      FM_database=database.data_remove_duplicates().TforO()
      FM_database=database.inputchar(typea)
      FMucustom=custom.TforO()
      FMucustom=custom.inputchar(typea)

    else:
      FM_database=database.data_remove_duplicates().Tfor()
      FM_database=database.inputchar(typea)
      FMucustom=custom.Tfor()
      FMucustom=custom.inputchar(typea)
    numrd_database=database.numrdata()
    numrdcustom=custom.numrdata()
    database_Tcs=database.set_avTc()
    custom_Tcs=custom.critical_temps
    totalq=np.append(FM_database,FMucustom,axis=0)
    totalz=np.append(numrd_database,numrdcustom,axis=0)
    totalT=np.append(database_Tcs,custom_Tcs)
    self.totalq=totalq
    self.totalz=totalz
    self.totalT=totalT
    descriptor_trainA,descriptor_testA,descriptor_trainb,descriptor_testb, Tc_train,Tc_test = train_test_split(totalq,totalz,totalT,random_state=seed, train_size=split)
    self.descriptor_trainA=descriptor_trainA
    self.descriptor_testA=descriptor_testA
    self.descriptor_trainb=descriptor_trainb
    self.descriptor_testb=descriptor_testb
    self.Tc_train=Tc_train
    self.Tc_test=Tc_test
# for saving low val loss
    if func=="mixture":
    
      def mixture(y_true,y_pred):    
        return tf.keras.losses.huber(y_true,y_pred,delta=7)
      funct = mixture
    else:
      funct=func
    
    if modeltype=="predictive":    
      checkpoint = ModelCheckpoint('modelfile',initial_value_threshold=100, monitor='val_loss', verbose=1, \
                              save_best_only=True, save_weights_only=True, \
                              mode='auto', save_frequency=1)

      models.compile(
          loss=[funct], optimizer=keras.optimizers.Adam(learning_rate=0.00027),metrics=[coeff_determination,"mean_absolute_percentage_error"],jit_compile=True,)
      models.fit([descriptor_trainA,descriptor_trainb],Tc_train,
          batch_size=500,
          epochs=epochs,
          
          verbose=1,
          validation_data=([descriptor_testA, descriptor_testb], Tc_test),callbacks=[checkpoint],
          shuffle=True)
      models.load_weights(Pathv+'/modelfile')
    return models,optimalTarget(database=database,custom=custom,FM_database=FM_database,FMucustom=FMucustom,numrd_database=numrd_database,numrdcustom=numrdcustom,database_Tcs=database_Tcs,custom_Tcs=custom_Tcs)
  def predict_structure(self,model_k,S=[],structureFiles=True,typea="original",returninput=False):
    if structureFiles:
      predictions=FourierSt(self.delta,self.Z_atomic,self.Z_mass,self.r0,self.gamvar,self.syref,filenames=S,pr=self.pr)
    else:predictions=FourierSt(self.delta,self.Z_atomic,self.Z_mass,self.r0,self.gamvar,self.syref,structures=S,filenames=[],pr=self.pr)
      
    if self.pr:
      F_=predictions.TforO()
    else: 
      F_=predictions.Tfor()
    F_=predictions.inputchar(typea)
    n_=predictions.numrdata()
    compositions=predictions.get_material_compositions()
    print(np.shape(F_),np.shape(n_))
    predictTcs=model_k.predict([F_,np.array(n_)])
    ind=1
    for structures_Tcs in predictTcs:
      print(compositions[ind-1],f'structure: {ind} Tc:{structures_Tcs}')
      ind+=1
    if returninput:
      return predictTcs,[F_,np.array(n_)]
    else:
      return predictTcs
  def modelfromsample(self,modeli,seed=42,func="mixture",split=0.80,epochs=300):
    descriptor_trainA,descriptor_testA,descriptor_trainb,descriptor_testb, Tc_train,Tc_test = train_test_split(self.totalq,self.totalz,self.totalT,random_state=seed, train_size=split)

    if func=="mixture":
    
      def mixture(y_true,y_pred):     
        return tf.keras.losses.huber(y_true,y_pred,delta=7)
      funct = mixture
    else:
      funct=func
    checkpoint = ModelCheckpoint('modelfile',initial_value_threshold=100, monitor='val_loss', verbose=1, \
                              save_best_only=True, save_weights_only=True, \
                              mode='auto', save_frequency=1)

    modeli.compile(
          loss=[funct], optimizer=keras.optimizers.Adam(learning_rate=0.00027),metrics=[coeff_determination,"mean_absolute_percentage_error"],jit_compile=True,)
    modeli.fit([descriptor_trainA,descriptor_trainb],Tc_train,
          batch_size=500,
          epochs=epochs,
          
          verbose=1,
          validation_data=([descriptor_testA, descriptor_testb], Tc_test),callbacks=[checkpoint], 
          shuffle=True)
    modeli.load_weights(Pathv+"/modelfile")
    return modeli

  def OptTc(self,v,numd,zas,model,optmax,target=0,typea='original'):
    Lat=makeLattice(v)
    numrdvolume=Lat.volume
    crst=makeCrystal(lat=Lat,minimaObj=v,buffer=zas)
    print(crst)

    try:
      numrdspace_group=pymatgen.core.Structure.get_space_group_info(crst,symprec=0.07,angle_tolerance=5)[1]
    except TypeError:
      numrdspace_group=1
    if self.pr:
      csH=self.TforO()
    else:
      csH=self.Tfor()
    cshM=self.inputchar(typea=typea)
    print(numrdspace_group)
    numerical=np.append(v[:6],np.array([numrdvolume,numrdspace_group]))
    numerical=np.array([np.append(numerical,numd)])
    
    Tc= model.predict([cshM,numerical])[0]
    
    print(Tc)
    if optmax == True:
      out=-Tc
    else:
      out=(Tc-target)**2
    return out  
  

  def superconductoropt(self,model_k,S=None,structuresfiles=True,typea='original',lbounds=[[2,12],[2,12],[2,12],[60,120],[60,120],[60,120]],posbounds=[],optmax=True,target=0,option={'method':'nelder-mead','tol':1.5,'options':{'eps':0.0002,'maxfev':1000,'ftol':1.51,'xatol':1.1,'fatol':1.5,'gtol':0.001,'adaptive':False}}):
    if structuresfiles:
      predictions=FourierSt(self.delta,self.Z_atomic,self.Z_mass,self.r0,self.gamvar,self.syref,structures=None,filenames=S,pr=self.pr)
    else:
      predictions=FourierSt(self.delta,self.Z_atomic,self.Z_mass,self.r0,self.gamvar,self.syref,structures=S,filenames=[],pr=self.pr)
    
    numd_=predictions.numrdata()
    
    pos,lats,z,none_=F_T_o(predictions.structures[0],Z_atomic)
    poslin=np.reshape(pos,3*len(z))
    Initial_=np.append(numd_[0][:6],poslin)
    print(Initial_)

    if posbounds == []:
      bounds=lbounds[0:6]

      for atoms in range(len(z)):
        bounds.append([0,1])
        bounds.append([0,1])
        bounds.append([0,1])
    else:
      bounds=lbounds[0:6]
      bounds.append(posbounds)
    
    
    
    optimized=scipy.optimize.minimize(predictions.OptTc,x0=Initial_,args=(numd_[0][8:],z,model_k,optmax,target,typea),bounds=bounds,
        method=option['method'],tol=option['tol'],options=option['options']
        )
    optimizedcrystal=makeAndWriteCrystal(optimized.x,buffer=z,write_f=False)
    self.last_opt = {'structure':optimizedcrystal}
    return optimized
  def opt(self,S=None):
    predictions=FourierSt(self.delta,self.Z_atomic,self.Z_mass,self.r0,self.gamvar,self.syref,structures=None,filenames=S,pr=self.pr)

    numd_=predictions.numrdata()
    
    pos,lats,z,none_=F_T_o(predictions.structures[0],Z_atomic)
    poslin=np.reshape(pos,3*len(z))
    Initial_=np.append(numd_[0][:6],poslin)
    print(Initial_)
    return Initial_,z


def yesbestTcsb(datamat,allAvgTc,crtklv,weights,a,n,et):
  tcs1 = []
  StatTc={}
  print("metrics",et)
  for crkt in crtklv:
    StatTc[crkt]=np.array([])
  for ind in range(0,len(datamat)):
    comp1 = datamat[ind]['structure'].composition
    ncomp3 = comp1.reduced_formula
    abserr_=0.1
    huber_=(keras.losses.huber(tf.convert_to_tensor([n[ind]], dtype=tf.float32),tf.convert_to_tensor([a[ind]], dtype=tf.float32),delta=7).numpy() ) /(et[2]) - 0.25
    meanabs_=(abs(n[ind]-a[ind])/((1.1*(n[ind])+abserr_)))/(et[4]/100)#/(2*((et[3])))
    if ncomp3 in list(crtklv.keys()) :
      weights[ncomp3][0][len(StatTc[ncomp3])]=100*(1-cmath.e**(-(0.0*huber_+1*meanabs_)))
      print("composition",ncomp3,"structure",len(StatTc[ncomp3]),"predicted",a[ind],"target",n[ind],"rate:",weights[ncomp3][0][len(StatTc[ncomp3])],"huber",huber_,"meanabs",meanabs_)

      if weights[ncomp3][0][len(StatTc[ncomp3])]>random.randrange(0,100):
        poolint=np.append(crtklv[ncomp3],np.mean(crtklv[ncomp3]))
        Tcchoice=random.choice(poolint)
        StatTc[ncomp3]=np.append(StatTc[ncomp3],Tcchoice)
        tcs1.append(Tcchoice)
      else: 
        
        StatTc[ncomp3]=np.append(StatTc[ncomp3],weights[ncomp3][1][len(StatTc[ncomp3])])
    else:
      tcs1.append(allAvgTc[ncomp3])
  return weights,crtklv

class optimalTarget(FourierSt):
  def __init__(self,database,custom,FM_database,FMucustom,numrd_database,numrdcustom,database_Tcs,custom_Tcs):
    self.database=database
    self.custom=custom
    self.FM_database=FM_database
    self.FMucustom=FMucustom
    self.numrd_database=numrd_database
    self.numrdcustom=numrdcustom
    self.database_Tcs=database_Tcs
    self.custom_Tcs=custom_Tcs








# #© Copyright

