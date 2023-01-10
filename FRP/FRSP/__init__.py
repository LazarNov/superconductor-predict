

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
#from astropy.convolution import convolve_fft
import tqdm
from tqdm import tqdm
import cvnn.layers as c_l
from aflow.keywords import reset
from aflow import *
from numba import jit,vectorize, float64
from numpy import mean,std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score, RepeatedKFold,train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
#from google.colab import drive
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
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
Pathv='/mnt/c/Users/Lazar/Desktop/model/FRSP/FRSP' #os.environ.get('PYTHONPATH')+'python3.7/site_packages/FRSP/FRSP'
#importfromdrive()
# Some files for processing without running every time.
#ccdat = pd.read_csv('/content/drive/My Drive/unique_m1.csv') # Elemental contributions, compounds, and Tcs from supercon. 
ccdat = pd.read_csv(Pathv+'/unique_m1.csv') # Elemental contributions, compounds, and Tcs from supercon. 
# Table from Molecules2021,26, 8. https://dx.doi.org/10.3390/molecules26010008 as it was parsed.

#rrnames = np.load('/content/drive/My Drive/valmatnams.npy', allow_pickle=True) 
aarrstuff3_primitive = np.load(Pathv+"/aarrstuff3_primitive.npy", allow_pickle=True) #MaterialsProject structures found from supercon data table earlier.
#matdf = pd.read_excel("/content/drive/My Drive/MaterialNames.xlsx")
#allsctraindf = pd.read_csv("/content/drive/My Drive/train.csv") # Chemical composition-based data from Molecules2021,26, 8. https://dx.doi.org/10.3390/molecules26010008
#matnams = matdf.values

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

# def init_composition_array(materialsarray):
#   arrcompsf=[]
#   for indx in range(len(materialsarray)):
#     matcompt = chemparse.parse_formula(str(materialsarray[indx]['structure'].composition).replace(" ", ""))
#     arrcompsf.append(matcompt)
#     #print(matcompt)
#   return arrcompsf

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


# def check_for_dup(infoarray):
#   norepeatmaterials = []
#   extraposition = []
#   for matindexs in range(len(infoarray)):
#     if infoarray[matindexs]['material_id'] in norepeatmaterials:
#       extraposition.append(matindexs)
#       print(matindexs,infoarray[matindexs]['material_id'])
#     else:     
#       norepeatmaterials.append(infoarray[matindexs]['material_id'])
#   return extraposition

# def data_remove_duplicates(infoarray):
#   originalsize = np.arange(len(infoarray))
#   nonduplicatepos = []
#   extraposarr = check_for_dup(infoarray)
#   [nonduplicatepos.append(x) for x in originalsize if x not in extraposarr]

#   print(nonduplicatepos)

#   reducedarray = []
#   for unipos in nonduplicatepos:
#     reducedarray.append(infoarray[unipos])

#   return reducedarray

# def maxofeachele(matcompsarray, showele = False):

#   maxeles = {}

#   for element in elementslist:
#     maxeles[element] = 0

#   for materialcomp in matcompsarray:

#     for elekey in materialcomp.keys():
#       if materialcomp[elekey]> maxeles[elekey]:
#         maxeles[elekey] = materialcomp[elekey]
#       if materialcomp[elekey]>= 100:
#         if showele:
#           print(materialcomp,elekey,materialcomp[elekey])
#   return maxeles

# def init_element_position_cols(elementmaxs):
#   elecolname=[]

#   for element in elementslist:
#     for k in range(0,int(elementmaxs[element])):
#       for j in ["x","y","z"]:
#         elecolname.append(element + str(k + 1) + j )
#   return elecolname

# def supercon_to_materialsproject(superconmaterials, lenmatdfval, showprint = False):
#   import pymatgen as pmg
#   ccdatfunc= superconmaterials.copy()

#   for n1 in range(0,lenmatdfval):
#     if showprint:
#       print(n1)
#       print(chemparse.parse_formula(matdf.values[n1][0]))

#     normcf2=chemNorm(chemparse.parse_formula(matdf.values[n1][0]))
#     ncomp2=pmg.core.composition.Composition(normcf2)
#     mplook2= ncomp2.reduced_formula

#     if showprint:
#       print(mplook2)

#     ccdatfunc["material"][n1] = mplook2
#   return ccdatfunc


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

# # 
# def init_lattice_dict(elementcolumnnames):
#   global latticeparms
#   latticeparms = {}
#   latticekeys = ["a","b","c","A","B","C","vol"] #a b c, angles A B C, volume of cell

#   for lkeys in latticekeys:
#     latticeparms[lkeys] = 0
#   print(latticeparms)

#   return latticeparms
# def init_position_dict(elementcolumnnames):

#   global posattmatf
#   # initialize dictionary 
#   posattmatf = {} 
    
#   # iterating through the elements of list 
#   for colnam in elementcolumnnames: 
#       posattmatf[colnam] = -12000 #-1
#   print(posattmatf) #posattmat before
#   return posattmatf

# def getallvals(datamat): # posattmat
#   allfeats = []
#   for ind in range(0,len(datamat)):
#     featpos = posattmatf.copy() # for original one posattmat
#     featlat = latticeparms.copy()
#     matcomp3 = chemparse.parse_formula(str(datamat[ind]['structure'].composition).replace(" ", ""))
#     atomcounter = 0
#     for espec in matcomp3.keys():
#       for k in range(0,int(matcomp3[espec])):
#         coordcount = 0
#         for j in ["x","y","z"]:
#           if  str(datamat[ind]['structure'][atomcounter].species) != espec + "1":
#             print(str(datamat[ind]['structure'][atomcounter].species),espec + "1")#"they aren't in the same order")
#           featpos[espec + str(k + 1) + j] = datamat[ind]['structure'][atomcounter].coords[coordcount]
#           if espec == 'H' and k+1>4:
#             print(ind, matcomp3 ) # matcomp?
#           coordcount = coordcount + 1

#         atomcounter = atomcounter + 1

#     featlat["a"]=datamat[ind]['structure'].lattice.a
#     featlat["b"]=datamat[ind]['structure'].lattice.b
#     featlat["c"]=datamat[ind]['structure'].lattice.c
#     featlat["A"]=datamat[ind]['structure'].lattice.angles[0]
#     featlat["B"]=datamat[ind]['structure'].lattice.angles[1]
#     featlat["C"]=datamat[ind]['structure'].lattice.angles[2]
#     featlat["vol"]=datamat[ind]['structure'].lattice.volume
#     featpos.update(featlat)
#     allfeats.append(featpos)
#   return allfeats 
############################################
############################################
############################################
#This is for the original one
#LOOKS REDUNDANT, nevermind, its kind of not
# def getallAvgTc(posattmatf,latticeparms,ccdat2,averagedTcs):
#   totalfeatures1 = posattmatf.copy() # posattmat.copy()
#   totalfeatures1.update(latticeparms)
#   global allAvgTc
#   allAvgTc = {}
#   countern = 0
#   for cnames in ccdat2['material']:
#     allAvgTc[cnames] = ccdat2['critical_temp'][countern]
#     print(cnames,ccdat2['critical_temp'][countern])
#     countern = countern + 1


#   for key in averagedTcs.keys():
#     allAvgTc[key] = averagedTcs[key]
#     print(key,averagedTcs[key])
#   return allAvgTc

# def getTcs(datamat,allAvgTc):
#   tcs1 = []
#   for ind in range(0,len(datamat)):
#     #print()
#     comp1 = datamat[ind]['structure'].composition
#     ncomp3 = comp1.reduced_formula
#     #print(ncomp3)

#     if ncomp3 == 'H2S':
#       print(ncomp3)
#      #if datamat[ind]['material_id'] == 'mp-24201':
#       #  print('1')
#       #  tcs1.append(190)
#       #elif datamat[ind]['material_id'] == 'mp-697135':
#       #  print('2')
#       #  tcs1.append(60)
#       #else:
#       tcs1.append(allAvgTc[ncomp3])
#     else:
#       tcs1.append(allAvgTc[ncomp3]) #averagedTcs before when it was all of them
      
#   return tcs1

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


# #tcmodel1 = getTcs(aarrstuff3)
# #tcmodelreduced = getTcs(arrstuffred)
# # 
# def getTcsavg(datamat):
#   tcs1 = []
#   for ind in range(0,len(datamat)):
#     #print()
#     comp1 = datamat[ind]['structure'].composition
#     ncomp3 = comp1.reduced_formula
#     #print(ncomp3)

#     if ncomp3 == 'H2S':
#       print(ncomp3)
#       if datamat[ind]['material_id'] == 'mp-24201':
#         print('1')
#         # tcs1.append(190) #
#       elif datamat[ind]['material_id'] == 'mp-697135':
#         print('2')
#         # tcs1.append(60)
#       else:
#         tcs1.append(averagedTcs[ncomp3])
#     else:
#       tcs1.append(averagedTcs[ncomp3]) #averagedTcs before when it was all of them
      
#   return tcs1

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

# Mg2Tsc=np.load('HighMg-1-2Li2H87t.npy')
# Mg2Tsc =np.transpose(Mg2Tsc)[0]
# fig= plt.figure(figsize=(16,16))
# plt.imshow(Mg2Tsc,origin='lower')
# plt.colorbar()
# for j in np.arange(0,15):
#   for i in [0,7,8,9]:
#     if Mg2Tsc[i,j]>350:
#       print(elementslist[j]+"2" + elementslist[i]+"4")

# HighMg_min1_min2Li2H87structures=np.load('HighMg-1-2Li2H87.npy',allow_pickle=True)
#

# Mg2Tsc=np.load('Mg2Li-3-4-5-6.npy')
# Mg2Tsc =np.transpose(Mg2Tsc)[0]
# fig= plt.figure(figsize=(16,16))
# plt.imshow(Mg2Tsc,origin='lower')
# plt.colorbar()
# for j in np.arange(0,15):
#   for i in [0,7,8,9]:
#     if Mg2Tsc[i,j]>350:
#       print(elementslist[j]+"2" + elementslist[i]+"4")

#for a list of material names find the structures on MaterialsProject
# Q is a composition list of the superconductors
# def matpro(Q):
#   from pymatgen.ext.matproj import MPRester
#   data=[]
#   with MPRester("e5UCPphwxb1umIG1Gjl") as m: 
#     for mats in tqdm(Q):
#       print(mats)
#       data2 = m.get_data(mats, prop = 'structure')
#       data.append(data2)
#   return data
#import mp_api
from tqdm import tqdm
def matpro(Q):
  from mp_api.client import MPRester
  data=[]
  with MPRester("ImK1c5FcoBQ44NkZIYbsvZa16vI409pI") as m: 
    for mats in tqdm(Q):
      print(mats)
      data2 = m.query()
      data.append(data2)
  return data
#names=ccdat.material
#data=matpro(names)

# Code that calculates some composition feats   Molecules2021,26, 8. https://dx.doi.org/10.3390/molecules26010008
# The formulas are in the manuscript.# code that calculates the composition feats.
import rpy2.robjects
from rpy2.robjects import packages
from rpy2.robjects import pandas2ri


pandas2ri.activate()
utils = packages.importr('utils')
base =  packages.importr('base')
#utils.chooseCRANmirror()
utils.install_packages("CHNOSZ")
utils.install_packages("xgboost")
CHN=packages.importr("CHNOSZ")
xgb=packages.importr("xgboost")
#ccdat = pd.read_csv('/content/drive/My Drive/unique_m1.csv') # Elemental contributions, compounds, and Tcs from supercon. 

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

  #model = Flatten()(model)#(inputs)
  #model0 = Flatten()(inputs) # When testing low delta versions convolutional layer was skipped.
  #model=concatenate([model,model0])
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
  # models=Dense(256,activation="relu")(models)
  # models= BatchNormalization(center=True, scale=True)(models)
  # models=Dense(256,activation="relu")(models)
  # models= BatchNormalization(center=True, scale=True)(models)

  #models = Dropout(0.50)(models)    
  # models=Dense(128,activation="relu")(models)
  # models = BatchNormalization(center=True, scale=True)(models)
  # models = Dropout(0.50)(models)
  # models=Dense(128,activation="relu")(models)
  # models= BatchNormalization(center=True, scale=True)(models)
  # models = Dropout(0.50)(models)#  models = Dense(512,activation='relu')(models)
#  models = BatchNormalization(center=True, scale=True)(models)
#  models = Dropout(0.50)(models)
#  models = Dense(512,activation='relu')(models)
#  models = BatchNormalization(center=True, scale=True)(models)
#  models = Dropout(0.50)(models)
#  models = Dense(512,activation='relu')(models)
#  models = BatchNormalization(center=True, scale=True)(models)
#  models = Dropout(0.50)(models)
#  models = Dense(512,activation='relu')(models)
#  models = BatchNormalization(center=True, scale=True)(models)
#  models = Dropout(0.50)(models)
#  models = Dense(512,activation='relu')(models)
#  models = BatchNormalization(center=True, scale=True)(models)
#  models = Dropout(0.50)(models)  
#  models = Dense(512,activation='relu')(models)
#  models = BatchNormalization(center=True, scale=True)(models)
#  models = Dropout(0.50)(models)
#  models = Dense(512,activation='relu')(models)
#  models = BatchNormalization(center=True, scale=True)(models)
#  models = Dropout(0.50)(models)
#
  outs = Dense(1,activation='relu')(models)
  models = Model(inputs=[inputsA,inputsB],outputs=[outs])
  # models.compile(
  #         loss="mse", optimizer=keras.optimizers.Adam(learning_rate=0.0007),jit_compile=True,
  #     )
  # models.fit([descriptor_trainA,descriptor_trainb], Tc_train,
  #           batch_size=64,
  #           epochs=epochs,
            
  #           verbose=1,
  #           validation_data=([descriptor_testA, descriptor_testb], Tc_test),
  #           shuffle=True)
  return models

# elirang=np.empty(shape=(1,2))
# elirang=	np.append(elirang,[[190,	228]],axis=0)#	500
# elirang=	np.append(elirang,[[184,	220]],axis=0)#	500
# elirang=	np.append(elirang,[[218,	245]],axis=0)#	500
# elirang=	np.append(elirang,[[360,	402]],axis=0)#	500
# elirang=	np.append(elirang,[[110,	139]],axis=0)#	500
# elirang=	np.append(elirang,[[134,	167]],axis=0)#	500
# elirang=	np.append(elirang,[[235,	269]],axis=0)#	500
# elirang=	np.append(elirang,[[257,	287]],axis=0)#	500
# elirang=	np.append(elirang,[[212,	244]],axis=0)#	500
# elirang=	np.append(elirang,[[213,	246]],axis=0)#	500
# elirang=	np.append(elirang,[[76,	104]],axis=0)#	500
# elirang=	np.append(elirang,[[285,	319]],axis=0)#	500
# elirang=	np.append(elirang,[[141,	180]],axis=0)#	500
# elirang=	np.append(elirang,[[232,	270]],axis=0)#	500
# elirang=	np.append(elirang,[[91,	121]],axis=0)#	500

# elirang=	np.append(elirang,[[130,	161]],axis=0)#	300
# elirang=	np.append(elirang,[[177,	207]],axis=0)#	300
# elirang=	np.append(elirang,[[138,	164]],axis=0)#	300
# elirang=	np.append(elirang,[[91,	115]],axis=0)#	300
# elirang=	np.append(elirang,[[137,	165]],axis=0)#	300
# elirang=	np.append(elirang,[[212,	233]],axis=0)#	300
# elirang=	np.append(elirang,[[135,	161]],axis=0)#	300
# elirang=	np.append(elirang,[[271,	301]],axis=0)#	300
# elirang=	np.append(elirang,[[59,	84]],axis=0)#	300
# elirang=  np.append(elirang,[[261,	293]],axis=0)#	300
# elirang=	np.append(elirang,[[112,	140]],axis=0)#	300
# elirang=	np.append(elirang,[[167,	198]],axis=0)#	300
# elirang=	np.append(elirang,[[130,	163]],axis=0)#	300

# elirang=	np.append(elirang,[[24,	39]],axis=0)#	200
# elirang=	np.append(elirang,[[260,	288]],axis=0)#	200
# #elirang=	np.append(elirang,[[245,	280]],axis=0)#	200
# elirang=	np.append(elirang,[[112,	134]],axis=0)#	200
# elirang=  np.append(elirang,[[196,	219]],axis=0)#	200
# elirang=	np.append(elirang,[[152,	175]],axis=0)#	200
# elirang=	np.append(elirang,[[73,	94]],axis=0)#	200
# elirang=  np.append(elirang,[[58,	78]],axis=0)#	200
# #elirang=	np.append(elirang,[[169,	204]],axis=0)#	200
# elirang=	np.append(elirang,[[196,	224]],axis=0)#	200
# elirang=	np.append(elirang,[[129,	162]],axis=0)#	200
# #elirang=	np.append(elirang,[[42,	58]],axis=0)#	200
# elirang=	np.append(elirang,[[56,	75]],axis=0)#	200
# elirang=	np.append(elirang,[[105,	134]],axis=0)#	200

# elirang=	np.append(elirang,[[40,	55]],axis=0)#	100
# elirang=	np.append(elirang,[[248,	279]],axis=0)#	100
# elirang=	np.append(elirang,[[40,	58]],axis=0)#	100
# #elirang=	np.append(elirang,[[49,	69]],axis=0)#	100
# elirang=	np.append(elirang,[[216,	253]],axis=0)#	100
# elirang=	np.append(elirang,[[61,	75]],axis=0)#	100
# elirang=	np.append(elirang,[[134,	161]],axis=0)#	100
# elirang=	np.append(elirang,[[134,	157]],axis=0)#	100
# #elirang=	np.append(elirang,[[13,	20]],axis=0)#	100
# elirang=elirang[1:]
# superconductcs=[np.mean(eli_) for eli_ in elirang]
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
class Prediction(tf.keras.callbacks.Callback):    
  def on_epoch_end(self, epoch, logs={}):
    y_pred = self.model([cshM[:0],np.array(csHn)[:0]])
    print('prediction: at epoch:',y_pred, epoch)
def mixture(y_true,y_pred):     
        return (0.00000100*tf.keras.losses.mean_squared_error(y_true,y_pred))**0.5 + 1.5*tf.keras.losses.huber(y_true,y_pred,delta=7) + coeff_determination(y_true,y_pred)

from keras import backend as kb
def coeff_determination(y_true, y_pred): # coeffdet function use
    return -( 1 - kb.sum(kb.square( y_true-y_pred ))/(kb.sum(kb.square( y_true - kb.mean(y_true) ) ) + kb.epsilon()) )

  
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

  #a_=sum(E_[0])**-1
  #b_=sum(E_[1])**-1
  #c_=sum(E_[2])**-1
  
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

  return Fmat#returns a pymatgen Lattice object corresponding to the found minima's lattice parameters

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
# prompt: select 1 

  #aarrstuff3_primitive = np.load("/content/drive/My Drive/aarrstuff3_primitive.npy", allow_pickle=True) #MaterialsProject structures found from supercon data table earlier.

  def __init__(self,compositionsfrac=ccdat.material,critical_temps=ccdat.critical_temp):
    
    self.compositionsfrac=compositionsfrac
    self.critical_temps=critical_temps
    #print(self.compositionsfrac)
  
  
 
  def renamecomp(self):
    import pymatgen as pmg
    def chemNorm(cform):
      fracCheck = False
      for key in cform.keys():
        #cform[key] = round(cform[key], 6) #float("{:.6f}".format(cform[key]))
        if cform[key].is_integer() == False:
          fracCheck = True
      
      while(fracCheck):
        fracCheck = False
        for key in cform.keys():
          cform[key] = 100*cform[key]
          cform[key] = round(cform[key], 6) #float("{:.6f}".format(cform[key]))
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
    #ccdatfunc.compositionsfrac=materials
    ccdatfunc=dataframe(materials,self.critical_temps)
    self.compositionsfrac=materials
    return ccdatfunc
    # if showprint:
    #   print(mplook2)

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
    
      def mixture(y_true,y_pred):      #loss=tf.reduce_mean((tf.square(y_true-y_pred)  /y_true),axis=-1)
        return (0.00000100*tf.keras.losses.mean_squared_error(y_true,y_pred))**0.5 + 1.5*tf.keras.losses.huber(y_true,y_pred,delta=7) + coeff_determination(y_true,y_pred)
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
      models.load_weights("modelfile")
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
  def modelfromsample(self,modeli,seed=42,func="mixture",split=0.80,epochs=300):#,totalq=None,totalz=None,totalT=None):
    #if type(totalq)==NoneType or type(totalz)==None or type(totalT)==None:
      # totalq=self.totalq
      # totalz=self.totalz
      # totalT=self.totalT
    #else:
    #  descriptor_trainA,descriptor_testA,descriptor_trainb,descriptor_testb, Tc_train,Tc_test = train_test_split(totalq, totalz,totalT,random_state=seed, train_size=split)
       
    #predictions=FourierSt(self.delta,self.Z_atomic,self.Z_mass,self.r0,self.gamvar,self.syref,filenames=["C1S1H7"])
    #csh=predictions.Tfor()
    #csh=predictions.inputchar()
    #cshM
    descriptor_trainA,descriptor_testA,descriptor_trainb,descriptor_testb, Tc_train,Tc_test = train_test_split(self.totalq,self.totalz,self.totalT,random_state=seed, train_size=split)

    if func=="mixture":
    
      def mixture(y_true,y_pred):     
        return (0.00000100*tf.keras.losses.mean_squared_error(y_true,y_pred))**0.5 + 1.5*tf.keras.losses.huber(y_true,y_pred,delta=7) + coeff_determination(y_true,y_pred)
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
    modeli.load_weights("modelfile")
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

# HighTC=np.load('High_Tc1F3-3m.npy',allow_pickle= True)
# HighTC

"""# New Section

# New Section
"""

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
      # if np.min(crtklv[ncomp3])>=1.5*a[ind]:
       
        #print(ncomp3,np.min(crtklv[ncomp3]),a[ind],mean([a[ind],np.min(crtklv[ncomp3 ])]))
      #   crtklv[ncomp3]=np.append(crtklv[ncomp3],mean([np.min(crtklv[ncomp3]),a[ind]]))
      # if np.max(crtklv[ncomp3])<a[ind]*2./3.:
      #   crtklv[ncomp3]=np.append(crtklv[ncomp3],mean([np.max(crtklv[ncomp3]),a[ind]]))
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


  # def WTcModel(self,cchk,ModelS,cut=1,seed=42,split=0.80,epochs=300):
  #   model_evl=1000
  #   crtklv={}

  #   listmodval=[model_evl]
  #   crtklvt=self.database.varTcs(cut)[1]
  #   vars=self.database.get_varTc()

  #   #norepeatst = self.database.structures

  #   norepeatdata=self.database.structures
  #   for i in self.database.get_material_compositions():
  #     if vars[i]>cut:
  #       crtklv[i]=crtklvt[i]
  #   allAvgTc=self.database.get_Tcmean()
  #   FM=self.FM_database
  #   FMu=self.FMucustom
  #   numrd=np.array(self.numrd_database)
  #   numrdu=np.array(self.numrdcustom)
  #   # allATcsC=allAvgTc.copy()
  #   #varnorepeatd,crtklv,hightvarrepeat=varianceTc_remove(norepeatdata,varianceTc,1)
  #   norepeatmodeltcC, allATcsC, weights= bestTcs(norepeatdata,allAvgTc,crtklv)
  #   # norepeatmodeltc=getTcs(norepeatdata,allAvgTc)

  #   norepeatmodeltcC= np.array(norepeatmodeltcC)
  #   #descriptor_train,descriptor_test, Tc_train,Tc_test = train_test_split(Fm,norepeatmodeltcC,train_size=0.7)  
  #   model=ModelS    
  #   #model.build(input_shapes)
  #   model.compile(loss=[mixture] , jit_compile=True,
  #                 optimizer=tf.keras.optimizers.Adam(learning_rate=0.00027),
  #                 metrics=[coeff_determination,'mean_absolute_error','mean_absolute_percentage_error','mse','cosine_similarity'])
  #   model.save_weights('resets')
  #   norepeatmodeltcC=self.database_Tcs
  #   superconductcs=self.custom_Tcs
  #   descriptor_trainA,descriptor_testA,descriptor_trainb,descriptor_testb, Tc_train,Tc_test = train_test_split(np.append(FM,FMu,axis=0),np.append(numrd,numrdu,axis=0),np.append(norepeatmodeltcC,superconductcs),random_state=seed,train_size=split)
  #   for epocks in range(cchk):#epoc
  #     # for crtkp in crtklv:
  #     #   allATcsC[crtkp] = random.choice(crtklv[crtkp])

  #     #tcmodellA=getTcs(varnorepeatd,allATcsC)
  #     #tcmodelA=getTcs(hightvarrepeat),allATcsC)
      

  #     keras.backend.clear_session()
  #     gc.collect()
  #     checkpoint = ModelCheckpoint('modelfile',initial_value_threshold=55.50466, monitor='val_loss', verbose=1, \
  #                             save_best_only=True, save_weights_only=True, \
  #                             mode='auto', save_frequency=1)

  #     model.compile(loss=[mixture] ,loss_weights=[1.0], jit_compile=True,
  #                 optimizer=tf.keras.optimizers.Adam(learning_rate=0.00027),
  #                 metrics=[coeff_determination,keras.losses.Huber(delta=7),'mean_absolute_error','mean_absolute_percentage_error',"log_cosh","mean_squared_logarithmic_error",'mse','cosine_similarity'])
  #     model.load_weights('resets')
      
  #     model.fit([descriptor_trainA,descriptor_trainb],Tc_train , 
  #               batch_size=500,
  #               epochs=epochs,
  #               verbose=1,
  #               validation_data= ([descriptor_testA,descriptor_testb],Tc_test),callbacks=[checkpoint])
      
  #     model.load_weights("modelfile")
  #     modevl=model.evaluate([descriptor_testA,descriptor_testb],Tc_test)[0]
  #     print(model.evaluate([descriptor_testA,descriptor_testb],Tc_test))
  #     listmodval.append(modevl)
  #     if modevl<model_evl:
  #       Top_allTcsC= allATcsC.copy()
  #       model_evl  = modevl
  #       Bestoh= norepeatmodeltcC.copy()
  #     weights,crtklv=yesbestTcsb(norepeatdata,allAvgTc,crtklv,weights,model.predict([FM,numrd]),n=norepeatmodeltcC,et=model.evaluate([descriptor_testA,descriptor_testb],Tc_test) )
  #     #for wgT in list(weights.keys()):
  #       #print(weights[wgT][0])
  #         # if weights[wgT][0]>=-40:
  #     norepeatmodeltcC,allATcsC,weights=bestTcsb(norepeatdata,allAvgTc,crtklv,weights)
  #     #for Hay in list(weights.keys()):
  #       #print(Hay,crtklv[Hay],weights[Hay][1],allAvgTc[Hay])

  #     fig=plt.figure(figsize=(15,15))
  #     plt.scatter(np.array(Tc_test),model.predict([descriptor_testA,descriptor_testb]))
  #     plt.show()
  #     norepeatmodeltcC= np.array(norepeatmodeltcC)
  #     descriptor_trainA,descriptor_testA,descriptor_trainb,descriptor_testb, Tc_train,Tc_test = train_test_split(np.append(FM,FMu,axis=0),np.append(numrd,numrdu,axis=0),np.append(norepeatmodeltcC,superconductcs),random_state=seed,train_size=split)
  #     print(epocks,model_evl)#,models.predict([cshM[1:],np.array(csHn)[1:]]))
      
  #     #model.compile(loss=mixture, jit_compile=True,
  #     #            optimizer=tf.keras.optimizers.Adam(learning_rate=0.00027),
  #     #            metrics=[coeff_determination,'mean_absolute_error','mean_absolute_percentage_error',"log_cosh","mean_squared_logarithmic_error",'mse','cosine_similarity'])
  #     model.load_weights('resets')



  #   return Top_allTcsC,model_evl,Bestoh

# np.load("Mg2Li-5.npy")

# """Predict with saved models try ms=FourierSt() modeltype=``predictivesave'' for saved predictive model"""
# ms=FourierSt()
# modelms=ms.samplemodels(modeltype='predictivesave',typea='original')

# """ms=FourierSt(pr=True) modeltype=``predictivesavepr'' for saved predictive model with more accuracte gaussians"""
# ms=FourierSt(pr=True)
# modelms=ms.samplemodels(modeltype='predictivesavepr',typea='original')

# ''' P-4m2 Pnma P3m1 Cm CmCm P-43m'''
# ms.predict_structure(modelms,['BC7.cif','AgTe.cif','SiC.cif','LiMg t).cif','CaGa.cif','ScZnCu3Se4.cif'])

# ms.predict_structure(modelms,["lah10_c2m.cif","yh10_r3m.cif","lah10_fm3m.cif","yh10_fm3m.cif","lah10_p63mmc.cif","yh10_cmcm.cif","lah10_r3m.cif","yh10_p63mmc.cif" ])

# ms.predict_structure(modelms,["YH10.cif",'LaH10.cif',"YH3_C2m_20GPa_CJP.cif",'YH3_Fm-3m.cif',"YH3_Fm-3m_0GPa_Kamegawa.cif","YH3_Fm-3m_130GPa_MIE.cif","YH3_I4mmm_50GPa_CJP.cif","YH3_P-3c1_ambientVS.cif","YH4_AO_I4mmm.cif","YH4_I4mmm_150GPa_AO.cif",'YH4_I4mmm_300GPa_CJP.cif','YH6_Im-3m.cif',"YH6_Im-3m_150GPa_AO.cif",'YH6_Im-3m_200GPa_CJP.cif',"YH7_Imm2.cif",'YH7_Imm2.cif',"YH7_P1.cif",'YH8_Fm-3m.cif',"YH9_Cc.cif",'YH9_F43m.cif','YH9_P63m.cif','YH9_P63mmc.cif',"250GPaFm-3mLaH10 .cif"])

# pymatgen.core.Structure.from_file(".cif",primitive=True)

# """ For predicting Tc from a list [cifs]"""
# ms.predict_structure(modelms,['I43m.cif'])
# """ For predicting Tc from a list of pymatgen structures [pmgs]"""
# ms.predict_structure(modelms,ms.data_remove_duplicates().structures[0:10],structureFiles=False)

# ms.predict_structure(modelms,["150GPaFm-3mLaH10 .cif"])

# ms.predict_structure(modelms,["YH6_Im-3m_200GPa_CJP - Copy.cif"])

# d=ms.data_remove_duplicates().structures
# ms.predict_structure(modelms,filenames,structureFiles=True)

# structure=pymatgen.core.structure.IStructure.from_file(filename='200GPa_C2_m_LaH7.cif',primitive=True)
# structure=pymatgen.core.structure.IStructure.from_file(filename='200GPa_Fm-3m_MgH13.cif',primitive=True)
# structure=pymatgen.core.structure.IStructure.from_file(filename=filenames[5+6-1-2-1+1-2],primitive=True)
# #pymatgen.io.cif.CifWriter(structure                                       ).write_file('C4S4.cif')

# LIfiles=["Li 2 MgH16 C2(Slash)m (300 GPa).cif","Li 2 MgH16 Cm (300 GPa).cif","Li 2 MgH16 Cmc2 (300 GPa).cif","Li 2 MgH16 Fd-3m (250 GPa).cif","Li 2 MgH16 I4 (300 GPa).cif","Li 2 MgH16 P-3m1 (300 GPa).cif","Li 2 MgH16 P1 (300 GPa).cif","Li 2 MgH16 Pm (300 GPa).cif","Li 2 MgH16 R32 (300 GPa).cif","Li 2 MgH16 α−P-1 (300 GPa).cif","LiMgH10 R-3m (300 GPa).cif","LiMgH14 Cc (300 GPa).cif","Li 2 MgH16 P-3m1 (300 GPa)(4).cif","TeH18(Br3O4)2.cif",'YH6_Im-3m_150GPa_AOFixed.cif']

# #structure=pymatgen.core.structure.IStructure.from_file(filename='200GPa_Fm-3m_MgH13.cif',primitive=True)
# #structure=pymatgen.core.structure.IStructure.from_file(filename=LIfiles[2 + 1],primitive=True)
# structure=pymatgen.core.structure.IStructure.from_file(filename='Li 2 MgH16 P-3m1 (300 GPa).cif',primitive=True)

# pymatgen.io.cif.CifWriter(structure).write_file('i434mCsh7.cif')
# pymatgen.io.cif.CifWriter(st[1]['structure']).write_file('MnClH36.cif')

# # structure=pymatgen.core.structure.IStructure.from_file(filename='YH4_I4mmm_167GPa.cif',primitive=True)
# # structure=pymatgen.core.structure.IStructure.from_file(filename='100GPa_Im-3m_CaH6.cif',primitive=True)   #
# testings='Li 2 MgH16 Fd-3m (250 GPa).cif' #''300GPa_Im-3m_MgH6.cif'
# structure=pymatgen.core.structure.IStructure.from_file(filename=testings,primitive=True)

# structure=pymatgen.core.structure.IStructure.from_file(filename='EntryWithCollCode685532.cif',primitive=True)

# #structure=pymatgen.core.structure.IStructure.from_file(filename='BaPd2.cif',primitive=True)
# structure=pymatgen.core.structure.IStructure.from_file(filename='HgBa2Ca2Cu3O7.cif',primitive=True)
# structure=pymatgen.core.structure.IStructure.from_file(filename='YH9_F43m.cif',primitive=True)
# La_YHfiles=["lah10_c2m.cif","yh10_r3m.cif","lah10_fm3m.cif","yh10_fm3m.cif","lah10_p63mmc.cif","yh10_cmcm.cif","lah10_r3m.cif","yh10_p63mmc.cif" ]

# ms.predict_structure(modelms,["Li 2 MgH16 Fd-3m (300 GPa).cif"])

# ms.predict_structure(modelms,["YH6_Im-3m_201GPa_2att.cif"])

# import pymatgen.core
# structure=pymatgen.core.structure.IStructure.from_file(filename= 'Li 2 MgH16 P-3m1 (300 GPa).cif',primitive=True)
# pymatgen.io.cif.CifWriter(structure).write_file("Li2MgH16P-3m1_.cif")

# #"LaH10.cif"
# #structure=pymatgen.core.structure.IStructure.from_file(filename= 'Li 2 MgH16 P-3m1 (300 GPa).cif',primitive=True)
# # structure=pymatgen.core.structure.IStructure.from_file(filename="lah10_p63mmc.vasp",primitive=True)

# #structure.remove_oxidation_states()

# structure
# #remove_site_property('oxidation_states')

# ms.predict_structure(modelms,[{'structure':structure}],False)
# #ms.predict_structure(modelms,["YH6_Im-3m_150GPa_AOFixed.cif"])

# # ms.predict_structure(modelms,["5677Li 2 MgH16 Fd-3m (250 GPa).cif"])
# # structure=pymatgen.core.structure.IStructure.from_file(filename="5677Li 2 MgH16 Fd-3m (250 GPa).cif",primitive=True)
# # structure.lattice.from_parameters(0.9,0.9,0.9,60,60,60)
# ms.predict_structure(modelms,["YH6_p63mmc_201.cif","YH6_p63mmc_237.cif"])
# structure

# fileZ = ms.predict_structure(modelms,LIfiles[:12])
# for A in range(12):
#   print(LIfiles[A],fileZ[A])

# HighTCey=[]
# for x in HighTC:
#   HighTCey.append({'structure':x})
# ms.predict_structure(modelms,HighTCey,False)

# startingmaterial='BaPd2.cif'
# '5777Li 2 MgH16 Fd-3m (250 GPa).cif'#HgBa2Ca2Cu3O7.cif'#filenames[15]
# startingmaterial='Li 2 MgH16 Fd-3m (250 GPa).cif'#'HgBa2Ca2Cu3O7.cif'#'200GPa_Fm-3m_MgH13.cif'
# startingmaterial=testings#'200GPa_R-3m_Mg2H5.cif'#'100GPa_Im-3m_CaH6.cif'#'Li 2 MgH16 Fd-3m (250 GPa).cif'#'Li 2 MgH16 P-3m1 (300 GPa).cif'Li 2 MgH16 Fd-3m (250 GPa).cif'#'Li 2 MgH16 P-3m1 (300 GPa).cif'#'YH6_Im-3m_200GPa_CJP - Copy.cif'#'Li 2 MgH16 P-3m1 (300 GPa).cif'###"LaH10.cif"#LIfiles[3]
# f=FourierSt(structures=[startingmaterial])
# Ini_,zz=f.opt([startingmaterial])
# sta=[]
# tt=[]
# maxse=[]
# maxey=[]
# rangea=np.arange(1,2,1)
# rangea_=np.arange(1,2,1)
# crystals=[]
# for atom in rangea:#[3]range(1,3):
#   maxse=[]
  
#   for atom2 in rangea_:#range(12,14):
#     #sta=[]
#     #zz[0]=atom
#     # zz[1]=12
#     # zz[-1]=atom#atom
#     # zz[-2]=atom
#     # zz[-3]=atom2
#     # zz[-4]=atom2
#     # zz[-5]=atom2
#     # zz[-6]=atom2
#     #z[6]=atom #atom
#     #zz[1]=atom2 #atom2#13#np.random.randint(1,84) zz[1]=1#3  zz[7]=1 print(zz)
#     scaling_=np.arange(1.1,6.1,4.9)
#     for I in scaling_:
#       #_st=[]
#       # zz[-5]=I
#       # zz[-6]=I
#       # zz[-3]=I
#       # zz[-4]=I
#       #zz=[1,1,1,1,1,1,1,1,1,1,1,1]
#       zz[-1]=52
#       zz[-2]=7
#       zz[-3]=1
#       zz[-4]=1
#       zz[-5]=1
#       zz[-6]=1
#       zz[0]=atom2#atom2#atom
#       #zz[1]=atom2#atom#*structure.lattice.b/structure.lattice.a #for t in range(60,120,2):
      
#       Ini_[0]=I#5.2145#I
#       Ini_[1]=I*structure.lattice.b/structure.lattice.a
#       Ini_[2]=I*structure.lattice.c/structure.lattice.a
#       #Ini_[3]=90.0
#       #Ini_[4]=75.5215
#       #Ini_[5]=90.0
#       csyt=makeAndWriteCrystal(Ini_,str(elementslist[0])+str(elementslist[0])  + startingmaterial,buffer=zz,write_f=False)
#       sta.append({'structure':csyt})
#       #maxey.append(csyt.get_space_group_info()[1])#try:
#       #   
#       #except TypeError:
#       #  maxey
#         # try:
#         #   numrdspace_group=pymatgen.core.Structure.get_space_group_info(csyt,symprec=0.07,angle_tolerance=5)[1]
#         # except TypeError:
#         #   numrdspace_group=1
        
#         # Ini_[6]=csyt.lattice.volume
#         # Ini_[7]=numrdspace_group
#       #STRUCz.lattice
#       #st.append(_st)
#       #st

#     #  tt.append(ms.predict_structure(modelms,_st,False))
# t2=ms.predict_structure(modelms,sta,False)
# st=np.reshape(t2,(len(rangea),len(rangea_),len(scaling_)))

# #maxey=np.reshape(maxey,((len(rangea)),len(rangea)))
# tmax=max(t2)
# #maxse.append(tmax)
# #tt.append(t2)
# #print(maxse)
#       #crystals.append(csyt)
#       # print(csyt.composition,max(maxse),'K')
#     #if np.isnan(t2[0]):
#     #  continue
#     #  print("Hy")
#   #maxey.append(maxse)
# # print(maxey)
# # print(max(maxey))

# tmax
# st[0][:]

# star =np.reshape(sta,(len(rangea),len(rangea_)))

# from scipy.signal.waveforms import nan
# #star_={}
# for i_0 in np.arange(0,len(st[:,0]),1):
#   for r_0 in np.arange(0,len(st[0,:]),1):
#     #print(nan)
#    # if st[i_0,r_0][0]==np.float32:
#       #print(st[0,1][0])
#     #star_[star[i_0,r_0]['structure'].composition.reduced_formula]=st[i_0,r_0]#
#     star

# starsort=sorted(star_.items(), key=lambda x: np.nan_to_num(x[1]),reverse=True)
# #starsort1=sorted(starsort, key=lambda x: float(x[1]),reverse=True)
# #maxey
# datast=pd.DataFrame(starsort)
# datast.to_csv('superconductors_m5.csv')
# #for i in range(len(starsort)):
# #  print(pd.DataFrame(starsort).values[i][0],pd.DataFrame(starsort).values[i][1],",")
# #(starsort[0][1])

# from matplotlib.colors import LogNorm
# xyt=np.transpose((t2))
# fig, ax = plt.subplots(figsize=(12,12))
# plt.title("T$_c$ of R-3m X$_2$$H$_{5}$",fontsize=18)
# plt.xticks(np.arange(1,87,5),fontsize=15)
# plt.yticks(np.arange(1,87,5),fontsize=15)
# plt.minorticks_on()
# plt.xlabel('X Atomic Number',fontsize=18)
# plt.ylabel('Y Atomic Number',fontsize=18)
# plt.imshow(np.reshape(st,(len(rangea),len(rangea))),cmap='rainbow',origin='bottom',interpolation="none",extent=[1,86,1,86])
# plt.colorbar()

# from matplotlib.colors import LogNorm
# fig, ax = plt.subplots(figsize=(12,12))
# plt.title("T$_c$ of Fd-3m XYH$_{36}$",fontsize=15)
# plt.xticks(np.arange(1,87,5),fontsize=15)
# plt.yticks(np.arange(1,87,5),fontsize=15)
# plt.minorticks_on()
# plt.xlabel('X Atomic Number',fontsize=15)
# plt.ylabel('Y Atomic Number',fontsize=15)
# plt.imshow((Mg2Tsc),cmap='rainbow',origin='bottom',vmin=250,interpolation="none",extent=[1,86,1,86])


# plt.colorbar()

# np.save("X2Y4H32",t)

# np.save("Mg0H1H12",t)

# fig, ax = plt.subplots(figsize=(12,12))
# plt.title("T$_c$ of Fd-3m Li$_{2}$MgH$_{16}$ K",fontsize=15)
# plt.yticks(ticks=np.arange(0,len(rangea),5),labels=np.arange(0.8,7.1,0.5),fontsize=15)
# plt.xticks(ticks=np.arange(0,len(rangea),5),labels=np.arange(0.8,7.1,0.5),fontsize=15)
# plt.xlabel('a $\AA$',fontsize=15)
# plt.ylabel('c $\AA$ ',fontsize=15)
# plt.imshow(st[0],origin='bottom',vmin=150,vmax=300,interpolation="spline16")
# plt.colorbar()
# # for i in tt[19:]:
# #   plt.plot(np.arange(1,40,1),i)#np.arange(0.8,6.1,0.1),i)
# # #plt.plot(np.arange(0.9,5.80,0.1),t2)

# fig, ax = plt.subplots(figsize=(12,12))
# plt.title("T$_c$ of P-3m1 Li$_{2}$MgH$_{16}$ K",fontsize=15)
# plt.yticks(ticks=np.arange(0,len(rangea),5),labels=np.arange(0.8,14.1,1),fontsize=15)
# plt.xticks(ticks=np.arange(0,len(rangea),5),labels=np.arange(0.8,14.1,1),fontsize=15)
# plt.xlabel('a $\AA$',fontsize=15)
# plt.ylabel('c $\AA$ ',fontsize=15)
# plt.imshow((tsd[0]),origin='bottom',vmin=200,vmax=300,interpolation="spline36")
# plt.colorbar()
# # for i in tt[19:]:
# #   plt.plot(np.arange(1,40,1),i)#np.arange(0.8,6.1,0.1),i)
# # #plt.plot(np.arange(0.9,5.80,0.1),t2)

# fig, ax = plt.subplots(figsize=(12,12))
# plt.title("T$_c$ of P-3m1 Li$_{2}$MgH$_{16}$ K",fontsize=15)
# #plt.yticks(ticks=np.arange(0,len(rangea),5),labels=np.arange(0.8,14.1,1),fontsize=15)
# #plt.xticks(ticks=np.arange(0,len(rangea),5),labels=np.arange(0.8,14.1,1),fontsize=15)
# plt.xlabel('a $\AA$',fontsize=15)
# plt.ylabel('c $\AA$ ',fontsize=15)
# plt.imshow(st[0],origin='bottom',vmin=200,vmax=300,interpolation="spline36")
# plt.show()
# plt.imshow(maxey,origin='bottom')
# (maxey)
# plt.colorbar()
# # for i in tt[19:]:
# #   plt.plot(np.arange(1,40,1),i)#np.arange(0.8,6.1,0.1),i)
# # #plt.plot(np.arange(0.9,5.80,0.1),t2)

# fig, ax = plt.subplots(figsize=(7,7))
# #plt.title("T$_c$ of P-3m1 Li$_{2}$MgH$_{16}$ K",fontsize=15)
# plt.yticks(ticks=np.arange(1,8,1),labels=np.arange(1,8,1),fontsize=16)
# plt.xticks(ticks=np.arange(1,8,1),labels=np.arange(1,8,1),fontsize=16)
# plt.xlabel('a ($\AA$)',fontsize=18)
# plt.ylabel('c ($\AA$) ',fontsize=18)
# stcop=st.copy()
# #for s in np.arange(len(st[0][0])-1):
# #  stcop[0][s][s]=st[0][s][s+1]
# plt.minorticks_on()

# plt.imshow((st[0]),origin='bottom',vmin=150,vmax=300,interpolation= "kaiser",aspect="equal",extent=(0.8,7,0.8,7))

# plt.colorbar(shrink=0.9).ax.tick_params(labelsize=14)
# #np.save('P-3m1acplot',st)
# #np.save('Fd-3m1acplot0',st)

# fig, ax = plt.subplots(figsize=(7,7))
# #plt.title("T$_c$ of P-3m1 Li$_{2}$MgH$_{16}$ K",fontsize=15)
# plt.yticks(ticks=np.arange(1,8,1),labels=np.arange(1,8,1),fontsize=16)
# plt.xticks(ticks=np.arange(1,8,1),labels=np.arange(1,8,1),fontsize=16)
# plt.xlabel('a ($\AA$)',fontsize=18)
# plt.ylabel('c ($\AA$) ',fontsize=18)
# stcop=st.copy()
# #for s in np.arange(len(st[0][0])-1):
# #  stcop[0][s][s]=st[0][s][s+1]
# plt.minorticks_on()

# plt.imshow((st[0]),origin='bottom',vmin=150,vmax=300,interpolation= "kaiser",aspect="equal",extent=(0.8,7,0.8,7))

# plt.colorbar(shrink=0.9).ax.tick_params(labelsize=14)
# #np.save('P-3m1acplot',st)
# #np.save('Fd-3m1acplot0',st)

# fig, ax = plt.subplots(figsize=(7,7))
# #plt.title("T$_c$ of P-3m1 Li$_{2}$MgH$_{16}$ K",fontsize=15)np.round(np.arange(1.1,14.1,2.4),3)
# plt.yticks(ticks=np.arange(1,15,1),labels=np.arange(1,15,1),fontsize=16)
# plt.xticks(ticks=np.arange(1,15,1),labels=np.arange(1,15,1),fontsize=16)
# plt.xlabel('a ($\AA$)',fontsize=18)
# plt.ylabel('c ($\AA$) ',fontsize=18)
# plt.minorticks_on()
# stcop=st.copy()
# #for s in np.arange(len(st[0][0])-1):
# #  stcop[0][s][s]=st[0][s][s+1]
# stcop=np.load('P-3m1acplot.npy')
# plt.imshow((stcop[0]),origin='bottom',vmin=150,vmax=300,interpolation="quadric",extent=(0.8,14,0.8,14),aspect="equal")

# plt.colorbar(shrink=0.9).ax.tick_params(labelsize=14)
# #np.save('P-3m1acplot',st)
# # np.save('Fs-3m1acplot',st)

# fig, ax = plt.subplots(figsize=(12,12))
# plt.title("T$_c$ of Fm-3m XYH$_{12}$",fontsize=15)
# plt.xticks(np.arange(1,87,5),fontsize=15)
# plt.yticks(np.arange(1,87,5),fontsize=15)
# plt.minorticks_on()
# plt.xlabel('X Atomic Number',fontsize=15)
# plt.ylabel('Y Atomic Number',fontsize=15)
# plt.plot(np.arange(0.8,6.1,0.1),t2)

# fig, ax = plt.subplots(figsize=(7,7))
# #plt.title("T$_c$ of P-3m1 LiMg$_{2}$H$_{16}$",fontsize=18)
# plt.minorticks_on()
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.xlabel('a ($\AA$)',fontsize=22)
# plt.ylabel('T$_c$ (K)',fontsize=22)


# plt.plot(rangea,np.transpose(t2)[0],markersize=12,linewidth=3,color='g')

# # fig, ax = plt.subplots(figsize=(8,8))
# # plt.title("T$_c$ of Fd-3m LiMg$_{2}$H$_{16}$",fontsize=18)
# # plt.minorticks_on()
# # plt.xticks(np.arange(0,7.51,0.5),fontsize=15)
# # plt.yticks(fontsize=15)
# # plt.xlabel('a $\AA$',fontsize=18)
# # plt.ylabel('T$_c$ (K)',fontsize=18)


# # plt.plot(rangea,np.transpose(t2)[0])
# fig, ax = plt.subplots(figsize=(7,7))
# #plt.title("T$_c$ of P-3m1 LiMg$_{2}$H$_{16}$",fontsize=18)
# plt.minorticks_on()
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.xlabel('a ($\AA$)',fontsize=22)
# #plt.ylabel('T$_c$ (K)',fontsize=22)


# plt.plot(rangea,np.transpose(t2)[0],markersize=12,linewidth=3,color='g')

# #pip install bayesian-optimization

# from bayes_opt import BayesianOptimization
# startingmaterial='BaPd2.cif'
# startingmaterial='200GPa_Fm-3m_MgH13.cif'
# startingmaterial=LIfiles[3]
# f=FourierSt(structures=[startingmaterial])
# Ini_,zz=f.opt([startingmaterial])
# st=[]
# ttone=[]
# maxseone=[]
# maxeyone=[]
# print(zz)
# #zz[0]=20
# #zz[1]=29
# #zz[2:]=[8,8,8,8,8,8,8,8,8,8,8]
# #zz[5]=29
# #Ini_[3:6]=[60,60,60]
# crystals=[]
# st=[]
# N=np.arange(1,87,1)
# M=np.arange(1,87)
# O=np.arange(1.1,1.2,0.1)
# P=[6,16]#np.arange(1,17,5)
# Q=[6,16]#np.arange(1,17,5)
# S=[6,16]#np.arange(6,17,10)
# for atom in N:#[3]range(1,3):
#   maxseone=[]
  
#   for atom2 in M:#range(12,14):
    
   
#     #zz[3]=atom2
#     #zz[16]=atom2
#     #zz[0]=atom
    
#     #zz[1]=atom#13#np.random.randint(1,84) zz[1]=1#3  zz[7]=1 print(zz)
#     #zz[2]=atom2
#     for I in O:
#       #_st=[]
#       # for atom3 in P:
#       #   for atom4 in Q:
#       #     for atom5 in S:
#             zz[-1]=atom
#             zz[-2]=atom2
#             zz[-3]=1
#             zz[-4]=1
#             zz[-5]=1
#             zz[-6]=1

#             Ini_[0]=I#*I/12
#             Ini_[1]=I
#             Ini_[2]=I#*I/12#I*structure.lattice.b/structure.lattice.a 
        
#         #for t in range(60,120,2):
#         #Ini_[2]=atom2#I*structure.lattice.c/structure.lattice.a
    
#             csyt=makeAndWriteCrystal(Ini_,buffer=zz,write_f=False
#                                     )
#             st.append({'structure':csyt})
#       #print(csyt)
#         # try:
#         #   numrdspace_group=pymatgen.core.Structure.get_space_group_info(csyt,symprec=0.07,angle_tolerance=5)[1]
#         # except TypeError:
#         #   numrdspace_group=1
        
#         # Ini_[6]=csyt.lattice.volume
#         # Ini_[7]=numrdspace_group
#       #STRUCz.lattice
#       #st.append(_st)
#       #st
    
#     #  tt.append(ms.predict_structure(modelms,_st,False))
# t=ms.predict_structure(modelms,st,False)
# T=np.reshape(t,(len(N),len(M),len(O)))
# tmax=max(t)
# maxseone.append(tmax)
# ttone.append(tmax)
#     #print(maxseone)
#      #crystals.append(csyt)
#     # print(csyt.composition,max(maxse),'K')
# if np.isnan(t[0]):
#   #continue
#   print("Hy")
# maxeyone.append(maxseone)

# print(maxeyone)
# print(max(maxeyone))

# np.save('HighMg-1-2Li2H87',st)
# np.save('HighMg-1-2Li2H87t',T)
# st[1]['structure']

# fig = plt.figure(figsize=(15,17))
# plt.imshow(np.max(T,axis=2),origin='lower')
# plt.colorbar()

# startingmaterial=LIfiles[3]
# f=FourierSt(structures=[startingmaterial])
# Ini_,zz=f.opt([startingmaterial])
# st=[]
# ttone=[]
# csyt=[]
# def morphfunction(z1,z2,z3,z4 ):
  
#   zz[-1]=int(round(z1))
#   zz[-2]=int(round(z2))
#   zz[-3]=int(round(z3))
#   zz[-4]=int(round(z3))
#   zz[-5]=int(round(z4))
#   zz[-6]=int(round(z4))
#   #print(csyt)
#         # Ini_[0]=I#*I/12
#         # Ini_[1]=I
#         # Ini_[2]=I#*I/12#I*structure.lattice.b/structure.lattice.a 
        
#         #for t in range(60,120,2):
#         #Ini_[2]=atom2#I*structure.lattice.c/structure.lattice.a
    
#   csyt=makeAndWriteCrystal(Ini_,buffer=zz,write_f=False
#                                     )
#   #st.append({'structure':csyt})
#       #print(csyt)
#         # try:
#         #   numrdspace_group=pymatgen.core.Structure.get_space_group_info(csyt,symprec=0.07,angle_tolerance=5)[1]
#         # except TypeError:
#         #   numrdspace_group=1
        
#         # Ini_[6]=csyt.lattice.volume
#         # Ini_[7]=numrdspace_group
#       #STRUCz.lattice
#       #st.append(_st)
#       #st
    
#     #  tt.append(ms.predict_structure(modelms,_st,False))
#   t=ms.predict_structure(modelms,[{'structure':csyt}],False)
#   if np.isnan(t):
#     t=0
#   else:
#     t=t[0][0]
#   #print(t)
#   return t

# optimizersres=[{'target': 350.19781494140625, 'params': {'z1': 57.85508226884797, 'z2': 8.254779589065958, 'z3': 7.885505410859069}},
# {'target': 352.0252380371094, 'params': {'z1': 56.50133771080748, 'z2': 8.086054086489796, 'z3': 7.828836318147408}},
# {'target': 348.12738037109375, 'params': {'z1': 62.60825365407608, 'z2': 8.259488561854964, 'z3': 8.223035145208613}},
# {'target': 352.0252380371094, 'params': {'z1': 57.448700210805804, 'z2': 7.987604994525456, 'z3': 8.012827732732655}},
# {'target': 350.9504089355469, 'params': {'z1': 61.14544223704435, 'z2': 7.911358570624015, 'z3': 1.3072279895619339}},
# {'target': 347.8241271972656, 'params': {'z1': 62.33454553641375, 'z2': 7.285632297746254, 'z3': 1.4034392941645315}},
# {'target': 349.1658020019531, 'params': {'z1': 57.736987368051466, 'z2': 7.008283754489482, 'z3': 6.660860479665789}},
# {'target': 350.9504089355469, 'params': {'z1': 61.157528018117205, 'z2': 8.157164698830535, 'z3': 1.1300209170217808}},
# {'target': 348.12738037109375, 'params': {'z1': 63.45729538946967, 'z2': 7.631132752310199, 'z3': 7.776819875039605}},
# {'target': 348.7565612792969, 'params': {'z1': 56.5899229554941, 'z2': 6.681736457045688, 'z3': 6.965035454283185}},
# {'target': 352.0252380371094, 'params': {'z1': 57.20033047811136, 'z2': 8.218393402789445, 'z3': 7.919214510769784}},
# {'target': 348.7565612792969, 'params': {'z1': 57.24845982975805, 'z2': 7.001808083509052, 'z3': 6.958488946357703}},
# {'target': 352.0252380371094, 'params': {'z1': 56.99472571221015, 'z2': 7.759624685922555, 'z3': 8.211174925858593}}]

# for s in optimizer.res:
#   if s['target']>=340:
#     optimizersres.append(s)
#     print(s)
# optimizer.max

# from bayes_opt import SequentialDomainReductionTransformer
# from bayes_opt.util import load_logs
# from bayes_opt.logger import JSONLogger
# from bayes_opt.event import Events

# logger = JSONLogger(path="./logs.json")
# bounds_transformer = SequentialDomainReductionTransformer(0.45,3,
#                                                           eta=0.96)
# optimizer = BayesianOptimization(f=morphfunction,pbounds={'z1':(1,86),'z2':(1,86),'z3':(1,17),'z4':(1,17)})
# optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
# optimizer.probe({'z1':12,'z2':12,'z3':3,'z4':3})
# optimizer.maximize(init_points=0, n_iter=150)

# load_logs(optimizer, logs=["./logs.json"]);

# optimizersres

# import ortools
# solver = pywraplp.Solver.CreateSolver('SCIP')


# z1 = solver.IntVar(1, 86, 'z1')
# z2 = solver.IntVar(1, 86, 'z2')
# z3= solver.IntVar(1, 17, 'z3')
# z4= solver.IntVar(1, 17, 'z4')
# z5= solver.IntVar(1, 17, 'z5')
# z6= solver.IntVar(1, 17, 'z6')

# optimizer.space
# for tareg in optimizer.res:
#   if tareg['target']>=345:
#     print(tareg)

# """{'target': 350.19781494140625, 'params': {'z1': 57.85508226884797, 'z2': 8.254779589065958, 'z3': 7.885505410859069}}
# {'target': 352.0252380371094, 'params': {'z1': 56.50133771080748, 'z2': 8.086054086489796, 'z3': 7.828836318147408}}
# {'target': 348.12738037109375, 'params': {'z1': 62.60825365407608, 'z2': 8.259488561854964, 'z3': 8.223035145208613}}
# {'target': 352.0252380371094, 'params': {'z1': 57.448700210805804, 'z2': 7.987604994525456, 'z3': 8.012827732732655}}
# {'target': 350.9504089355469, 'params': {'z1': 61.14544223704435, 'z2': 7.911358570624015, 'z3': 1.3072279895619339}}
# {'target': 347.8241271972656, 'params': {'z1': 62.33454553641375, 'z2': 7.285632297746254, 'z3': 1.4034392941645315}}
# {'target': 349.1658020019531, 'params': {'z1': 57.736987368051466, 'z2': 7.008283754489482, 'z3': 6.660860479665789}}
# {'target': 350.9504089355469, 'params': {'z1': 61.157528018117205, 'z2': 8.157164698830535, 'z3': 1.1300209170217808}}
# {'target': 348.12738037109375, 'params': {'z1': 63.45729538946967, 'z2': 7.631132752310199, 'z3': 7.776819875039605}}
# {'target': 348.7565612792969, 'params': {'z1': 56.5899229554941, 'z2': 6.681736457045688, 'z3': 6.965035454283185}}
# {'target': 352.0252380371094, 'params': {'z1': 57.20033047811136, 'z2': 8.218393402789445, 'z3': 7.919214510769784}}
# {'target': 348.7565612792969, 'params': {'z1': 57.24845982975805, 'z2': 7.001808083509052, 'z3': 6.958488946357703}}
# {'target': 352.0252380371094, 'params': {'z1': 56.99472571221015, 'z2': 7.759624685922555, 'z3': 8.211174925858593}}
# """

# np.save('HighMg-1-2Li2H',st)
# fig = plt.figure(figsize=(15,17))
# plt.imshow(np.max(T,axis=2),origin='lower')
# plt.colorbar()

# np.save('HighMg-1-2Li2Hr2',st)
# fig = plt.figure(figsize=(15,17))
# plt.imshow(np.max(T,axis=2),origin='lower')
# plt.colorbar()

# np.save('HighMg-1-2Li2Hr3',st)
# fig = plt.figure(figsize=(15,17))
# plt.imshow(np.max(T,axis=2),origin='lower')
# plt.colorbar()

# a= np.transpose(np.array(maxeyone))[0]
# l = np.array([len(a[i]) for i in range(len(a))])
# width = l.max()
# b=[]
# for i in range(len(a)):
#     if len(a[i]) != width:
#         x = np.pad(a[i], (0,width-len(a[i])), 'constant',constant_values = 0)
#     else:
#         x = a[i]
#     b.append(x)
# b = np.array(b)
# #print(b)
# a=[]
# i=0
# for A in np.transpose(np.array(maxeyone)):
#   #print(A)
  
#   for x in range(i):
#     A.append([0])
#   a.append(A)
#   i+=1
# print(a)

# np.max(T,axis=2)

# fig = plt.figure(figsize=(15,17))
# xaxis=elementslist[:len(N)]
# plt.imshow(np.max(T,axis=2),origin='lower',)
# plt.colorbar()
# fig = plt.figure(figsize=(15,15))
# #ax = plt.axes(projection='3d')
# plt.plot(N ,np.max(T,axis=2)[:,2], color='blue')
# print(np.nanmax(T))
# ax.set_title('wireframe');

# np.save("Mg2Li-3-4-5-6",T)

# import matplotlib
# fig = plt.figure(figsize=(15,17))
# xaxis=elementslist[:len(N)]
# plt.imshow(np.max(T,axis=2),origin='lower',vmax=300)
# plt.colorbar()
# fig = plt.figure(figsize=(15,15))
# ax = plt.axes(projection='3d')
# ax.contour(N,M, np.max(T,axis=2),50,stride=0.05, color='blue')
# ax.set_title('wireframe');
# ax.view_init(60,35)

# plt.imshow(np.transpose(maxeyone)[0]) 
# plt.colorbar()

# plt.imshow(np.transpose(maxeyone)[0]) 
# plt.colorbar()

# max(tt)

# zz[0]=25
# zz[1]=1

# syt=makeAndWriteCrystal(Ini_,buffer=zz,write_f=True)
# syt

# np.save("Li2MgH16Tcinitial",maxey)
# np.save("Li2MgH16listallTcs",tt)

# np.save('High_Tc_H24-3',crysts)

# def anglesfunc(Init):
#   Ini_[3:6]=Init
#   csyt=makeAndWriteCrystal(Ini_,buffer=zz,write_f=False)
#   try: prediction=ms.predict_structure(modelms,[{'structure':csyt}],False)[0]
#   except TypeError:
#     prediction=0
#   print(Init)
#   return -prediction
# #for i in np.arange(1,260):
# #  print(anglesfunc([i,i,i]))
# #anglesfunc([60,60,60])
# print(Ini_)
# def latticefunc(Init):
#   Ini_[0:6]=Init
#   csyt=makeAndWriteCrystal(Ini_,buffer=zz,write_f=False)
#   try: prediction=ms.predict_structure(modelms,[{'structure':csyt}],False)[0]
#   except TypeError:
#     prediction=0
#   print(Init)
#   return -prediction
#   Ini_[:6]=[3.        ,  3.        ,  3.        , 59.77961986, 59.77971643,
#        59.82426139]
# scipy.optimize.minimize(latticefunc,Ini_[:6],bounds=[(1,4),(1,4),(1,4),(59,65),(59,65),(59,65)],method='nelder-mead',options={'eps':0.1})

# t

# figr=plt.figure(figsize=(12,9))
# #plt.plot(np.arange(1.2,1.45,0.1),t)
# #figr=plt.figure(figsize=(12,9))
# plt.plot(np.arange(1,70,1),np.transpose(maxey)[0][0])
# plt.figure(figsize=(18,18))
# plt.plot([1.2 for x in np.arange(2,202,0.5)],np.arange(2,402))
# plt.imshow(np.transpose(maxey)[0])
# plt.colorbar()
# [elementslist[x]+'H13' for x in[11, 12, 21, 22, 30, 31, 32, 33, 36, 50, 51, 52, 54]]
# atoms1=[1,3,4,11,12,13,21,22,30,31,32,33,36,50,51,52,54]
# atoms2=[11,12,21,22,30,31,32,33,36,50,51,52,54]
# crysts=[]
# for x in np.arange(0,0):
#   for y in np.arange(0,69):
#     print(maxey[x][y][0])
#     if maxey[x][y][0]>0:
#       #zz[0]=x+1
#       zz[0]=y+1
#       print(elementslist[x]+elementslist[y],maxey[x][y])
#       csyt=makeAndWriteCrystal(Ini_,buffer=zz,write_f=False)
#       crysts.append(csyt)

# for i in range(len(maxey)):
#   for j in range(len(maxey[i])):
#     if maxey[i][j][0]>220:
#       zz[-1]=i+1
#       zz[-2]=j+1
#       makeAndWriteCrystal(Ini_,fileName= str(i+1) +" " + str(j+1)  + 'xyLi2H32-1.cif',buffer=zz,write_f=True)
#       print(maxey[i][j][0])
# #csyt=makeAndWriteCrystal(Ini_,buffer=zz,write_f=False)
# #crysts.append(csyt)
# mx= np.array(maxey)
# plt.imshow(np.transpose(np.array(maxey))[0])
# plt.colorbar()

# from numpy.ma.core import sort

# [x for x in np.arange(len(maxse)) if maxse[x]>330]

# plt.imshow(np.transpose(maxey)[0])
# plt.colorbar()

# #plt.imshow(tuple(np.array(tt[:][:][0])))
# #test = np.random.rand(10,12)
# #print(np.shape(test))
# # plt.imshow(test)
# #plt.show()
# #np.shape(np.transpose(np.array(tt[:][:][:]),(2,0,1))[0])

# plt.imshow(np.transpose(np.array(tt[:][:][:]),(2,0,1))[0],cmap='ocean')
# plt.colorbar()
# #print(tt)

# """ optimizing for maximum Tc """
# ms.superconductoropt(modelms,['C1S1H7.cif'])
# """ optimizing for pickedTc """
# # ms.superconductoropt(modelms,['C1S1H7.cif'],optmax=False,target=180)
# """ Specify bounds and optimization latparamaters angles"""
# ms.superconductoropt(modelms,['C1S1H7.cif'],lbounds=[[2,12],[2,12],[2,12],[60,120],[60,120],[60,120]],option={'method':'nelder-mead','tol':0.1,'options':{'maxfev':3000}})

# """ Show last optimization """
# ms.last_opt
# """ Start optimization from last structure"""
# minimaobj=ms.superconductoropt(modelms,[ms.last_opt],structuresfiles=False)
# """ make cif and crystal with Z= specifying atomic numbers """
# makeAndWriteCrystal(minimaobj.x,fileName="optout.cif",buffer=[6,1,1,1,1,1,1,1,16])
# """ make cif from last optimization"""
# pymatgen.io.cif.CifWriter(ms.last_opt['structure']).write_file('optout.cif')

# """ Generate and fit sample model and data """
# F=FourierSt(pr=True)
# modeli,outputs=F.buildsample(modeltype='predictivesavepr',typea='original')

# """ Build and fit prebuilt model build sample data """
# F=FourierSt(pr=True)
# outputs=F.sampledata(typea='original')

# """ Build and fit prebuilt model or build sample data """
# modesl=F.samplemodels(modeltype='predictive',typea='original')
# modesl=F.modelfromsample(modesl)

# #make database structures 

# database=FourierSt(structures=aarrstuff3_primitive,filenames=[],compositionsfrac=ccdat.material,critical_temps=ccdat.critical_temp,pr=True)
# custom = FourierSt(filenames=filenames,critical_temps=superconductcs,pr=True)

# FM_database=database.data_remove_duplicates().TforO()
# FM_database=database.inputchar(typea='original')
# FMucustom=custom.TforO()
# FMucustom=custom.inputchar(typea='original')

# numrd_database=database.numrdata()
# numrdcustom=custom.numrdata()
# database_Tcs=database.set_avTc()
# custom_Tcs=custom.critical_temps

# ms=FourierSt()
# r2=ms.samplemodels(modeltype='r2model',typea='original')

# r2,datasets=ms.buildsample('r2model')

# prS=ms.samplemodels("predictivesavepr","original")

# r2pred=np.transpose(r2.predict([ms.descriptor_testA,ms.descriptor_testb]))
# from sklearn.metrics import r2_score
# r2_score(ms.Tc_test,r2pred[0])

# figure=plt.figure(figsize=(7,7))
# plt.xticks()
# figure.add_axes()
# figure.align_labels()
# plt.minorticks_on
# plt.xticks(fontsize=18)
# #plt.xlim(right=300)
# plt.yticks(fontsize=18)
# line=np.arange(0,250,1)
# plt.title('$T_c$  test vs. $T_c$ predicted ',fontsize=23)
# plt.xlabel("T$_c$ predicted (K)",fontsize=19)
# plt.ylabel("T$_c$ (K)",fontsize=19)
# plt.scatter(r2.predict([ms.descriptor_testA,ms.descriptor_testb]),ms.Tc_test,color='r')
# plt.plot(line,line,'--',color='teal',label="$T_c$  = $T_c$ predicted")
# plt.legend(fontsize=12)
# plt.show()


# datapredictive=FourierSt(pr=True)
# datapredictive.totalq=np.append(FM_database,FMucustom,axis=0)
# datapredictive.totalz=np.append(numrd_database,numrdcustom,axis=0)
# datapredictive.totalT=np.append(database_Tcs,custom_Tcs)

# model_predictive_s=datapredictive.samplemodels('predictive','original')
# model_predictive_s=datapredictive.modelfromsample(model_predictive_s)
# dfda,nmda=datapredictive.predict_structure(model_predictive_s,["C1S1H7.cif"],typea="original")
# dfda,nmda=datapredictive.predict_structure(model_predictive_s,aarrstuff3_primitive[19:23],structureFiles=False,typea="original")
# SC_O=datapredictive.superconductoropt(model_predictive_s,['C1S1H7.cif'],typea="original")

# F_set=F.sampledata(typea="original")
# sam=F.samplemodels('predictive','original')
# model_i=F.modelfromsample(sam,epochs=300)

# ""
# d = dataframe()
# ms.compositionsfrac
# d=d.renamecomp()

# d.compositionsfrac

# from IPython.display import Latex
# from pandas.io.formats.style_render import DataFrame
# superd=DataFrame({'Composition':d.compositionsfrac,'Tc':d.critical_temps})
# TcVec=[]
# for i in range(5):
#   TcVec.append(d.get_Tc_ve()[d.compositionsfrac[i]])

# tcDataframe=DataFrame({'Comp':d.get_Tc_ve().keys(),'Tcs':d.get_Tc_ve().values(),'Average(Tc)':d.get_Tcmean().values(),('Var(Tc)'):d.get_varTc().values()})
# tcDataframe

# superd

# FourierSt().data_remove_duplicates().structures

# # Search Aflow for data. Structures from comp.
# # ccdat 
# def aflowmat(ccdat):
#   structures=([])
#   for a1 in range(len(ccdat)):
#     kspcs=",".join(list(chemparse.parse_formula(ccdat['material'][a1]).keys()))
#   # for string in list(chemparse.parse_formula(ccdat['material'][1]).keys()):
#   #   kspcs += string


#     kstch=",".join([str(round(el/sum(list(chemparse.parse_formula(ccdat['material'][a1]).values())),8)) for el in list(chemparse.parse_formula(ccdat['material'][a1]).values())])
#     reset()

#     k= ((K.species ==kspcs) & (K.stoichiometry ==kstch) & (K.nspecies==len(chemparse.parse_formula(ccdat['material'][a1]))))#str(round(2.0/3.0,9))+","+str(round(1.0/3.0,9))))
#     result = search().filter(k)
#     reset()
#     if len(result)>0: 
#       for results in result:
      
#         structure = [pymatgen.core.Structure.from_str(results.files[results.prototype[:-1]+".cif"](), fmt='cif', primitive=True)]
#         structures.append(structure)
#   return structures

# aowl={} 
# ()

# from keras.utils.generic_utils import default
# # For tuning the convolutional layers. Tried several different config, depends on delta gamvar syref used.
# #from cvnn.activations import cart_relu,modrelu,zrelu
# import cvnn.activations as crl
# def buildmodelA(hp,inputs,input_d):
#   x=inputs

# #  cvnn.activations.cart_relu
#   #x = c_l.ComplexInput(input_shape=(input_d))
#   #for cvlayrs in range(hp.Int('cvl',2,2)):
#   #actk=hp.Choice('ac',values=['cart_relu1','zrelu1','modrelu1'],ordered=False)

#   #x = c_l.ComplexConv3D(32, kernel_size=(3), activation=actk[0:-1])(x)
#   x = c_l.ComplexConv3D(32, kernel_size=(hp.Int('ksize',3,5,2)), activation='cart_relu')(x)
#   x = (c_l.ComplexAvgPooling3D(pool_size=(2)))(x)
#   # x = c_l.ComplexBatchNormalization(center=True, scale=True)(x)
#   if hp.Boolean('c50'):
#     x=c_l.ComplexDropout(0.50)(x)
  
#   # x = c_l.ComplexConv3D(32, kernel_size=(3), activation=actk[0:-1])(x)
#   x = c_l.ComplexConv3D(hp.Int('kunitsiz',32,64,32), kernel_size=(3), activation='convert_to_real_with_abs')(x)
#   x = (c_l.ComplexAvgPooling3D(pool_size=(2)))(x)
#   # x = c_l.ComplexBatchNormalization(center=True, scale=True)(x)
#   if hp.Boolean('c50'):
#     x=c_l.ComplexDropout(0.50)(x)

#   x = c_l.ComplexFlatten()(x) 
#   #actlm=hp.Choice('adplm',values=['cart_relu1','modrelu1','zrelu1'],ordered=False)
#   # for clay in range(hp.Int('cdl',1,5,1)):
#   #   #x = c_l.ComplexDense(units=2**(hp.Int('out_unit',5,7)),activation=actlm[:-1])(x)
#   #   x = c_l.ComplexDense(units=2**(hp.Int('out_unit',4,8)),activation='zrelu')(x)

#   # x = c_l.ComplexDense(units=2**(hp.Int('outlayer',5,10)),activation='convert_to_real_with_abs')(x)

#     # if hp.Boolean("dropout"):
#     #   x=keras.layers.Dropout(0.50)(x)
#     # for i in range(hp.Int("cnn_layers", 0, 0,step=1)):
#     #     x = keras.layers.Conv3D(
#     #         hp.Int("filters_",32, 64, step= 32),
   
#     #         kernel_size=hp.Int("size", 3,3,1),
#     #         activation="relu")(x)
#     #     x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
#     #     x=BatchNormalization(center=True, scale=True)(x)
#     #     if hp.Boolean("dropout"):
#     #       x=keras.layers.Dropout(0.50)(x)
  
#   #x =keras.layers.Flatten()(x)
#   model =x 
#   return x#model

# # For tuning the layers for the numerical data. 
# # This is a range being tested exact values may not be restricted to this in other testing.
# def buildmodelB(hp,inputs,inputd):
#   model=inputs
#   # model =  Dense(units=2**hp.Int("units", min_value=5, max_value=10),input_dim=inputd,
#   # activation="relu")(inputs)
#   for i in range(hp.Int("numlayers",2,7 )):
#   #for i in range(hp.Int("numlayers",2,4 )):
#     model =  Dense(units=2**hp.Int("units", min_value=5, max_value=9),activation="relu")(model)
#     model =(BatchNormalization(center=True, scale=True))(model)

#   return model

# # plotting the reconstruction of the components. use if syref is True.
# # from matplotlib import projections

# # index=1456
# # domaine=np.arange(-(delta-1) /2,(delta+1 )/2,1)
# # Fs=fS #fSu
# # def a(x,y,z,lata,latb,latc):
# #   T=0
# #   u1,v1,w1=-1,-1,-1

# #   for h in domaine:
# #     v1=-1
# #     u1+=1
# #     for k in domaine: 
# #       w1=-1
# #       v1+=1

# #       for l in domaine:    
# #         w1+=1

# #         T+=Fs[1][index][u1][v1][w1]*cmath.e**(2*np.pi*cmath.sqrt(-1)*(x*h/lata + y*k/latb + z*l/latc))

# #   return T
# # fig=plt.figure(figsize=(16,9))
# # x1,y1,z1=np.array([]),np.array([]),np.array([])
# # for X1 in np.arange(0,norepeatdata[0]['structure'].lattice.abc[0],0.35):
#   # for Y1 in np.arange(0,norepeatdata[0]['structure'].lattice.abc[1],0.35):
#     # for Z1 in np.arange(0,norepeatdata[0]['structure'].lattice.abc[2],0.35): 
#       # if a(X1,Y1,Z1,norepeatdata[0]['structure'].lattice.abc[0],norepeatdata[0]['structure'].lattice.abc[1],norepeatdata[0]['structure'].lattice.abc[2])>4:
#         # x1=np.append(x1,X1)
#         # y1=np.append(y1,Y1)
#         # z1=np.append(z1,Z1)

# # X, Y = np.meshgrid(np.arange(0,norepeatdata[index]['structure'].lattice.abc[0],0.1),np.arange(0,norepeatdata[index]['structure'].lattice.abc[1],0.1))

# # ax0 = fig.gca(projection='3d')


# # ax0.plot_surface(X,Y,a(X,Y,0,norepeatdata[index]['structure'].lattice.abc[0],norepeatdata[index]['structure'].lattice.abc[1],norepeatdata[index]['structure'].lattice.abc[2]))
# #Fig=plt.figure(figsize=(16,9))
# #Fig.gca(projection='3d').scatter3D(x1,y1,z1)
# # plt.show()
# # norepeatdata[1456]['structure']

# from keras.layers import concatenate
# # For tuning the concatinated 
# def buildM(hp,input_shapeA):
#   keras.backend.clear_session()
#   gc.collect()
#   #inputsA = keras.Input(shape=input_shapeA)
#   inputsA = c_l.complex_input(shape=input_shapeA)
#   inputsB = keras.Input(shape=len(numrd[1]))
#   modela=buildmodelA(hp,inputsA,input_shapeA)
#   modeln=buildmodelB(hp,inputsB,len(numrd[1]))
#   models=concatenate([modela,modeln])
#   #modela=createconvolutionlayers(inputsA,input_shapeA)  #modeln=createmlmaterial(inputsB,len(numrd[1]))

#   # for n in range(p=1)):
#   for n in range(hp.Int("layers",1,5,step=1)):
#     models = Dense(2**hp.Int(f"units_m",5,9,1),activation="relu")(models)
#     models = BatchNormalization(center = True, scale=True)(models)
#     if hp.Boolean("dropout"):      
#       models = Dropout(0.50)(models)
#   outs = Dense(1,activation='relu')(models)
#   models = Model(inputs=[inputsA,inputsB],outputs=[outs])

#   # msqe=hp.Float('msqe',0,3.1)
#   def mixture(y_true,y_pred):
#     return ((0.000000100*tf.keras.losses.mean_squared_error(y_true,y_pred))**0.1e+1 )+ 1*tf.keras.losses.huber(y_true,y_pred,delta=7) + coeff_determination(y_true,y_pred) + (12*0.*tf.keras.losses.mean_squared_logarithmic_error(y_true,y_pred)**2)
#   models.compile(loss=[mixture] ,loss_weights=[1.0], jit_compile=True,
#             optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
#             metrics=[ coeff_determination, 'mean_absolute_percentage_error',"log_cosh","mean_squared_logarithmic_error",'mse','cosine_similarity'])
#    # $\
#   return models

# def build_concat(hp,inputsA,inputsB,input_shapeA):
#   models=concatmodel(inputsA,inputsB,input_shapeA)
  
#   meansqe=hp.Float('meansqe',0.006,0.1)
#   #hd=hp.Float('hd',1,20)
  
#   def mixture(y_true,y_pred):
#     loss=mean(sum((y_true-y_pred)**2 /y_true))
#     return loss #meansqe*tf.keras.losses.mean_squared_error(y_true,y_pred) + tf.keras.losses.huber(y_true,y_pred,delta=7) 
#   models.compile(
#           loss=[mixture], optimizer=keras.optimizers.Adam(learning_rate=0.0007),metrics=[mixture,coeff_determination,"mse","mean_absolute_percentage_error"],jit_compile=True,
#       )
#   return models

# class Prediction(tf.keras.callbacks.Callback):    
#   def on_epoch_end(self, epoch, logs={}):
#     y_pred = self.model([cshM[1:],np.array(csHn)[1:]])
#     print('prediction: at epoch:',y_pred, epoch)

# def ModelOpt(epochs=9,max_trials=7,input_shapeA=input_shapeA,descriptor_trainA=descriptor_trainA,descriptor_trainb=descriptor_trainb,Tc_train=Tc_train,descriptor_testA=descriptor_testA,descriptor_testb=descriptor_testb,Tc_test=Tc_test,batch_size=None):
#   keras.backend.clear_session()
#   gc.collect()
#   tuner = kt.RandomSearch( 
#       hypermodel=partial(buildM,input_shapeA=input_shapeA),
#       objective=[kt.Objective("val_loss",direction="min")],
#       max_trials=max_trials,
#       executions_per_trial=1,
      
#       overwrite=False,
#       project_name="real",
#       )
#   tuner.search([descriptor_trainA,descriptor_trainb], Tc_train, epochs=epochs,batch_size=batch_size, callbacks=[keras.callbacks.TensorBoard("real/tb"),Prediction()],validation_data=([descriptor_testA,descriptor_testb], Tc_test))
#   bestm = tuner.get_best_models()[0]
#   tuner.results_summary(45)
#   return bestm,tuner

# def concatOpt(epochs=9,max_trials=7,input_shapeA=input_shapeA,descriptor_trainA=descriptor_trainA,descriptor_trainb=descriptor_trainb,Tc_train=Tc_train,descriptor_testA=descriptor_testA,descriptor_testb=descriptor_testb,Tc_test=Tc_test,batch_size=None):
#   keras.backend.clear_session()
#   gc.collect()
#   tuner = kt.RandomSearch( 
#       hypermodel=partial(build_concat,inputsA=inputsA,inputsB=inputsB,input_shapeA=input_shapeA),
#       objective=[kt.Objective("val_coeff_determination",direction="min")],
#       max_trials=max_trials,
#       executions_per_trial=1,
#       overwrite=True,

#       project_name="ahyLOW",
#       )
#   tuner.search([descriptor_trainA,descriptor_trainb], Tc_train, epochs=epochs,batch_size=batch_size, callbacks=[keras.callbacks.TensorBoard("ahyLOW/tb")], validation_data=([descriptor_testA,descriptor_testb], Tc_test))
#   bestm = tuner.get_best_models()[0]
#   tuner.results_summary()
#   return bestm,tuner

# def theors(InitialIN,ts,Tbuffer):
#   Initialmattheor=np.array(InitialIN)
#   for theoats in range(ts):
#     Initialmattheor=np.append(Initialmattheor,np.array([random.random(),random.random(),random.random()]))
#   crystalStruct = makeCrystal(makeLattice(Initialmattheor),Initialmattheor,buffer=Tbuffer)
#   comp1 = crystalStruct.composition
#   ncomp3 = comp1.reduced_formula
#   feattheoretical=np.array(Extract(ncomp3,ss_e_d))
#   for i in range(len(elementslist)):
#     feattheoretical = np.append(feattheoretical,comp1[i+1])
#   return feattheoretical, Initialmattheor

# ""

# ("model.predict(Fmatsplit_High),getTcs(norepeatd_High)")

# #models=concatmodel(inputsA,inputsB,input_shapeA,epochs=270)
# # Sample for testing variance weights.
# def gamvaroptmize(gamvar):

#   fS=Tfor(delta=13,gamvar=[gamvar[0],gamvar[1]],syref=False)

#   FM=inputchar(fS,typea="abs")
#   descriptor_trainA,descriptor_testA,descriptor_trainb,descriptor_testb, Tc_train,Tc_test = train_test_split(np.append(FM,FMu,axis=0),np.append(numrd,numrdu,axis=0),np.append(S,superconductcs),random_state=42, train_size=0.80)
#   fltrs=np.shape(FM)[4]
#   delta=np.shape(FM)[3]
#   input_shapeA=(delta, delta, delta,fltrs)
#   inputsA = keras.Input(shape=input_shapeA)
#   inputsB = keras.Input(shape=len(numrd[1]))
#   checkpoint = ModelCheckpoint('modelfile',initial_value_threshold=60., monitor='val_loss', verbose=1, \
#                              save_best_only=True, save_weights_only=False, \
#                              mode='auto', save_frequency=1)

#   models=concatmodel(inputsA,inputsB,input_shapeA,epochs=270)

#   models.compile(
#          loss=mixture, optimizer=keras.optimizers.Adam(learning_rate=0.0007),metrics=["mean_absolute_percentage_error"],jit_compile=True,
#       )
#   models.load_weights("reset")
#   models.fit([descriptor_trainA,descriptor_trainb], Tc_train,
#           batch_size=128,
#               epochs=100,
        
#             verbose=1,
#             validation_data=([descriptor_testA, descriptor_testb], Tc_test),
#             shuffle=True,callbacks=[checkpoint])
#   models.load_weights("modelfile")
#   er= models.evaluate([descriptor_testA,descriptor_testb],Tc_test)[0]
#   shutil.rmtree("modelfile")
#   print(er,gamvar)
#   return er

# # This is testing

# # def SuperM(attemptsdata=30,iters=1):#prototype
# #   numrd=np.array(numrdata(norepeatdata,numdit))
# #   distancelowest=1000
# #   for iter in range(iters):
# #     distlooping=10000
# #     for loopingdata in range(attemptsdata):

# #       Gamvar1=5* 10**random.uniform(-7,-2)
# #       Gamvar2=10**random.uniform(-1,0)
# #       gamvar=[Gamvar2,Gamvar1]
# #       syref=random.choice([True,False])

# #       if syref==False: 
# #         delt=random.randint(a=10,b=20)
# #       else:
# #         delt=2*random.randint(a=6,b=16)+1

# #       fS=Tfor(delta=delt,gamvar=gamvar,syref=syref)
    
# #       for typeA in ["abs","complex"] :
# #         FM=inputchar(fS,typea=typeA)
# #         if iter==0:
# #           descriptor_trainA,descriptor_testA,descriptor_trainb,descriptor_testb, Tc_train,Tc_test = train_test_split(np.append(FM,FMu,axis=0),np.append(numrd,numrdu,axis=0),np.append(S,superconductcs),random_state=42, train_size=0.80)
# #           print(iter,delt)
# #         else:
# #           descriptor_trainA,descriptor_testA,descriptor_trainb,descriptor_testb, Tc_train,Tc_test = train_test_split(np.append(FM,FMu,axis=0),np.append(numrd,numrdu,axis=0),np.append(bests_lowest,superconductcs),random_state=42, train_size=0.80)

# #         fltrs=np.shape(FM)[4]
# #         delta=np.shape(FM)[3]
# #         input_shapeA=(delt, delt, delt,fltrs)
# #         inputsA = keras.Input(shape=input_shapeA)
# #         inputsB = keras.Input(shape=len(numrd[1]))

# #         ModelMS=ModelOpt(input_shapeA=input_shapeA,descriptor_trainA=descriptor_trainA,descriptor_trainb=descriptor_trainb,Tc_train=Tc_train,descriptor_testA=descriptor_testA,descriptor_testb=descriptor_testb,Tc_test=Tc_test)
# #         ModelMSone=ModelMS
# #         ModelMSone.build(input_shapeA)
# #         print("delta:",delta,input_shapeA)
# #         ModelMSone.compile(loss=mixture, optimizer=keras.optimizers.Adam(learning_rate=0.0007),metrics=["mean_absolute_percentage_error"],jit_compile=True,)
# #         ModelMSone.fit([descriptor_trainA,descriptor_trainb], Tc_train,
# #                   epochs=50,
# #                   batch_size=None,
# #                   verbose=1,
# #                   validation_data=([descriptor_testA, descriptor_testb], Tc_test),
# #                   shuffle=True)
# #         stA=ModelMSone.evaluate([descriptor_testA,descriptor_testb],Tc_test)[0]
                              
# #         if stA <= distlooping:
# #           FMS=FM
# #           ModelS=ModelMS
# #           input_shapeS=input_shapeA
# #           distlooping=stA
# #           G1zlowest=Gamvar1
# #           G2zlowest=Gamvar2
# #           deltazlowest=delt
# #           syrefzlowest=syref

# #           print("distlooping",distlooping,"G1zlowest",G1zlowest,"G2zlowest",G2zlowest,"deltazlowest",deltazlowest,"syrefzlowest",syrefzlowest)
  
# #   # ModelS.build(input_shapeS)
# #     ModelWon=ModelS
# #     ModelWon.build(input_shapeS)
# #     d,var,bestS=WtopTcModel(cchk=12,ModelS=ModelWon,input_shapes=input_shapeS,FM=FMS,numrd=numrd)
# #     ModelWon.compile(
# #             loss=mixture, optimizer=keras.optimizers.Adam(learning_rate=0.0007),metrics=["mean_absolute_percentage_error"],jit_compile=True,
# #                     )
    
# #     ModelWon.fit([FMS,numrd], bestS,
# #               epochs=30,
# #               batch_size=None,
# #               verbose=1,
# #               validation_data=([FMS, numrd], bestS),
# #               shuffle=True)
# #     st=ModelWon.evaluate([FMS,numrd],bestS)[0]
# #     fig=plt.figure(figsize=(18,16))
# #     plt.scatter(ModelWon.predict([FMS,numrd]),bestS)
# #     plt.show()
# #     if iter==0 or  st <=distancelowest:

# #       dlowest=d
# #       varlowest=var
# #       bests_lowest=bestS
# #       ModelMlowest=ModelS
# #       distancelowest =st
# #       G1lowest=G1zlowest
# #       G2lowest=G2zlowest
# #       deltalowest=deltazlowest
# #       syreflowest=syrefzlowest

# #       best_parameters={"varlowest":varlowest,"distancelowest":distancelowest,"G1lowest":G1lowest,"G2lowest":G2lowest,"deltalowest":deltalowest,"syreflowest":syreflowest}

# #       print("varlowest",varlowest,"distancelowest",distancelowest,"G1lowest",G1lowest,"G2lowest",G2lowest,"deltalowest",deltalowest,"syreflowest",syreflowest)
      
      
      


# #   return ModelMlowest,bests_lowest,dlowest,best_parameters

# #© Copyright

