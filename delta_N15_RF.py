from ncload import ncload
import numpy as np
from netCDF4 import Dataset
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import math
import datetime


# subroutine for the calculation of earth area
def pxlarea(lati,latr,lonr):
    # lati = upper initial lat
    # latr = lat span angle
    # lonr = lon span angle
    # sphere surface area: S=2pi*r*r*(1-sin(theta))
    r = 6371.0 # unit: km
    pppi = math.pi
    alfa = (math.sin(lati/180.*math.pi)-math.sin((lati-latr)/180.*math.pi))
    beta = np.float64(lonr)/360.
    cal_area0 = 2*math.pi*r**2*alfa*beta
    cal_area = abs(2.*math.pi*r**2*(math.sin(np.float64(lati)/180.*math.pi)-math.sin((np.float64(lati)-np.float64(latr))/180.*math.pi))*np.float64(lonr)/360.)
    return cal_area

# Calculate the grid area
lat_area = np.arange(-90.0,90.00,0.1)
lon_area = np.arange(0,360.,0.1)
grid_area = np.zeros(len(lat_area))
for ii in range(0,len(lat_area)):
    grid_area[ii] = pxlarea(lat_area[ii],-0.1,0.1)  # unit: km2
grid_area[-1] = 0.0


# set the lontitude and latitude used in this code
lon = np.arange(-180,180,0.1)
lat = np.arange(-56,84,0.1)

###################################################################
########################   Data loading ###########################

########### 1. load the Plant Functional Type (PFT) map
########### and the fraction of land in grids
# load the PFT map
pft_frac_nc = ncload('PFT_map_10th_deg.nc')
pft_frac = np.array(pft_frac_nc.get('pft_frac')[:].filled(np.nan))
mask_desert = (pft_frac[0]>=85)       # obtain a desert mask
pft_map_new = np.array(pft_frac[1:])  # exclude the desert
pft_map_new[9] = 0.0  # set the fraction of C3 pasture to zero
pft_map_new[11:14] = 0.0 # set the fraction of C4 pasture, C3-C4 crop area to zero

pft_sum = np.nansum(pft_map_new,axis=0) # the fraction of natural ecosystems (except desert)

# load the fraction of continential area
land_frac_nc = ncload('land_fraction_from_CRU_10th.nc')
land_frac_data = land_frac_nc.get('land_frac')
land_frac_data1 = land_frac_data[:]*grid_area[:,np.newaxis]*1e6  # land area
# land area of natural ecosystems
land_frac_new = np.array(land_frac_data1[340:1740].filled(np.nan))*pft_sum[340:1740]

# transpose the matrix, reshape it into a nrow*1 vect
land_frac_mat = np.transpose(np.flipud(land_frac_new),(1,0))
land_frac_vect = np.squeeze(land_frac_mat.reshape(-1,1))

land_area_new = land_frac_data1[:]*np.sum(pft_map_new[:],axis=0) # This is exactly the same as the matrix land_frac_mat

# mask the desert
land_area_new = np.where(np.sum(pft_frac[1:],axis=0)<=0.15,np.nan,land_area_new)
land_area_tailored = land_area_new[340:1740] # the land area used for computing the N fluxes

# check the area of natural ecosystems
land_area_natural = np.nansum(np.nansum(land_area_tailored,axis=0),axis=0)/1e12

############ 2. data loading for the RF algorithm

# data path for the 16 predictors
global_data_path = '*********'
# list of the file names for the 16 predictors
file_name_list = ['gpp1982_2015.nc','nhx_tian_2005_2014.nc','noy_tian_2005_2014.nc',\
                  'p_pet1980_2018.nc','pre1980_2018.nc','tmp1980_2018.nc',\
                  'BD.nc','CLAY.nc','SAND.nc','SILT.nc','OC.nc','TC.nc','TN.nc',\
                  'PHH2O.nc','am.nc','em.nc','nfix.nc']
# list of the variable names for the 16 predictors
var_name_list = ['gpp','nhx','noy','p_pet','pre','tmp',\
                 'BD','CLAY','SAND','SILT','OC','TC','TN','PHH2O',\
                 'am','em','nfix']

# load the .nc files, get the variable, and transform the matrices (lat*lon) into vects (nrow*1)
kk=0
for file_name in file_name_list:
    exec('file_p = ncload(global_data_path+"%s")'%file_name) # load the .nc file
    exec('%s = file_p.get("%s")'%(var_name_list[kk],'tmp'))  # get the variable
    exec('%s_mat = np.array(%s[:].filled(np.nan))'%(var_name_list[kk],var_name_list[kk]))
    exec('%s_vect = np.squeeze(%s_mat.reshape(-1,1))'%(var_name_list[kk],var_name_list[kk])) # reshape the matrix into vect
    kk = kk+1


############ 3. load the soil N15 observations
Craine_soil_data = 'soilP_v1_agg_tianC.xlsx'
soil_data = pd.read_excel(Craine_soil_data,header=0,index_col=0)

# drop all the nan/empty values
soil_new=soil_data.dropna()

# calculate the C/N ratio with total carbon (TC) and total nitrogen (TN)
soil_rf_tc = soil_new['TC'].values
soil_rf_tn = soil_new['TN'].values

soil_new['C/N'] = np.where(soil_rf_tn<1e-3,100000,soil_rf_tc/soil_rf_tn)

# keep all vects of the 16 predictors, and drop all the useless value
soil_rf = soil_new.drop(labels=['Latitude','Longitude','AWC_CLASS','TC','TN'],axis=1)
feature_labels = pd.Series(['BD','Clay','OC','pH','Sand','Silt','AM','EM','GPP','Nfix','NHx','NOy','P','T','P/ET','C/N'])

###################################################################################
############################# Training of the RF algorithm  #######################

# To prepare the training data for the RF algorithm
X = (soil_rf.drop(labels=['N15','pet'],axis=1)).values # Predictors
Y = (soil_rf['N15']).values  # N15 observations

# Establish the RF model
forest = RandomForestRegressor(n_estimators=500,random_state=0,bootstrap=True,oob_score=True,min_samples_leaf=1,max_features='sqrt',n_jobs=4)
# Fit the RF model
forest.fit(X,Y)


# The training and validation data
Ysim = forest.predict(X)
COMP_MAT = np.zeros((len(Ysim),3))
COMP_MAT[:,0] = Y
COMP_MAT[:,1] = Ysim
COMP_MAT[:,2] = forest.oob_prediction_
Train_Val = pd.DataFrame(COMP_MAT,columns=['observation','Train','OOB'])
Train_Val.to_excel('RF_soil_Train_validation.xlsx')

# Importance vect of the 16 predictors
importances = forest.feature_importances_

IMP = pd.DataFrame(importances,columns=['Importance'])
IMP.to_excel('RF_soil_Importance.xlsx')

index = np.argsort(importances)

######################################################################################
###########################  Predict the global map of soil delta_N15 ################

# X_global is the predictor vects across the globe
X_global = np.zeros((5040000,16))
#mask_true = (BD[:].filled(np.nan) == BD[:].filled(np.nan))
#mask_nan = np.logical_not(mask_true)

# Prepare the predictor vects
X_global[:,0] = BD_vect[:]
X_global[:,1] = CLAY_vect[:]
X_global[:,2] = OC_vect[:]
X_global[:,3] = PHH2O_vect[:]
X_global[:,4] = SAND_vect[:]
X_global[:,5] = SILT_vect[:]
X_global[:,6] = am_vect[:]
X_global[:,7] = em_vect[:]
X_global[:,8] = gpp_vect[:]
X_global[:,9] = nfix_vect[:]
X_global[:,10] = nhx_vect[:]
X_global[:,11] = noy_vect[:]
X_global[:,12] = pre_vect[:]
X_global[:,13] = tmp_vect[:]
X_global[:,14] = p_pet_vect[:]
X_global[:,15] = np.where(TN_vect[:]<1e-3,100000,TC_vect[:]/TN_vect[:])

# get the mask of nan values
mask_nanana = (X_global!=X_global)
# get the mask of infinite values
mask_inf = np.argwhere(np.isinf(X_global))
# set the nan values to a given value 0.5;
# this value is only used in very limited grids, which are excluded in the final map
X_global[mask_nanana] = 0.5
# set the inf values to a large value 100000
X_global[mask_inf] = 100000

# set the a zero vect of the global N15
N15_global = np.zeros((5040000,))
# Predict the global N15 using the well-trained RF
N15_global = forest.predict(X_global)
# Remove the grids which are nan in the predictors
N15_global[mask_nanana[:,1]] = np.nan

# Reshape the N15 map
N15_mat = N15_global.reshape(3600,-1)
N15_mat_new = np.transpose(np.fliplr(N15_mat),(1,0))
N15_mat_new[mask_desert] = np.nan # mask the desert

# Set the ensemble of global maps of soil N15, to calculate the SD of N15
N15_std_global_mat = np.zeros((5040000,len(forest.estimators_)))

kk=0
for t in forest.estimators_:
    print(kk)
    # predict the global soil N15 using each decision tree
    N15_std_global_mat[:,kk]=t.predict(X_global)
    kk=kk+1

# Compute the land area-weighted N15 for all ensembles
N15_mean_vect = np.nansum(N15_std_global_mat*land_frac_vect[:,np.newaxis],axis=0)/np.nansum(land_frac_vect)

# Compute the mean, SD, and quantiles (2.5%, 50%, 97.5%) of these global means
soil_N15_mean = np.nanmean(N15_mean_vect)
soil_N15_std = np.nanvar(N15_mean_vect)**0.5
soil_N15_mean_q025 = np.nanquantile(N15_mean_vect,0.025)
soil_N15_mean_q50 = np.nanquantile(N15_mean_vect,0.50)
soil_N15_mean_q975 = np.nanquantile(N15_mean_vect,0.975)

# Determine the SD and quantiles (2.5%, 50%, 97.5%) of the global N15 maps
N15_std_global = np.var(N15_std_global_mat,axis=1)**0.5
N15_q025_global = np.nanquantile(N15_std_global_mat,0.025,axis=1,interpolation='lower')
N15_q50_global = np.nanquantile(N15_std_global_mat,0.50,axis=1,interpolation='lower')
N15_q975_global = np.nanquantile(N15_std_global_mat,0.975,axis=1,interpolation='lower')

# Mask the grids with nan values in the predictor
N15_std_global[mask_nanana[:,1]] = np.nan
N15_q025_global[mask_nanana[:,1]] = np.nan
N15_q50_global[mask_nanana[:,1]] = np.nan
N15_q975_global[mask_nanana[:,1]] = np.nan

# Mask the grids with infinite large value
X_global[mask_inf] = np.nan
mask_large = (X_global==100000)
X_global[mask_large] = np.nan

# Reshape the global maps of SD, and quantiles of N15 maps
N15_std_mat = N15_std_global.reshape(3600,-1)
N15_std_mat_new = np.transpose(np.fliplr(N15_std_mat),(1,0))
N15_std_mat_new[mask_desert] = np.nan

N15_q025_mat = N15_q025_global.reshape(3600,-1)
N15_q025_mat_new = np.transpose(np.fliplr(N15_q025_mat),(1,0))
N15_q025_mat_new[mask_desert] = np.nan

N15_q50_mat = N15_q50_global.reshape(3600,-1)
N15_q50_mat_new = np.transpose(np.fliplr(N15_q50_mat),(1,0))
N15_q50_mat_new[mask_desert] = np.nan

N15_q975_mat = N15_q975_global.reshape(3600,-1)
N15_q975_mat_new = np.transpose(np.fliplr(N15_q975_mat),(1,0))
N15_q975_mat_new[mask_desert] = np.nan

########################################################################
############################## Data saving #############################
atts = dict(description = "Global maps of soil delta_N15",
            contact = "Shushi Peng (speng@pku.edu.cn), Maoyuan Feng (fengmy@pku.edu.cn)",
            date = datetime.datetime.now().strftime("%d/%m/%Y %H:%M"),
            resolution = "Regular 0.01-degree")
RF_file = Dataset('RF_soil_N15.nc','w')
RF_file.ncattrs(atts)
RF_file.createDimension('longitude',len(lon))
RF_file.createDimension('latitude',len(lat))
long = RF_file.createVariable('lon','d',('longitude',))
latit = RF_file.createVariable('lat','d',('latitude',))

long[:] = lon[:]
latit[:] = lat[:]

N15_lf = RF_file.createVariable('N15_soil','f',('longitude','latitude'))
N15_lf_std = RF_file.createVariable('N15_soil_std','f',('longitude','latitude'))
N15_lf_lowq = RF_file.createVariable('N15_soil_lowq','f',('longitude','latitude'))
N15_lf_uppq = RF_file.createVariable('N15_soil_uppq','f',('longitude','latitude'))
N15_lf_midd = RF_file.createVariable('N15_soil_midd','f',('longitude','latitude'))


N15_lf[:] = np.transpose(N15_mat_new[:],(1,0))
N15_lf_std[:] = np.transpose(N15_std_mat_new[:],(1,0))
N15_lf_lowq[:] = np.transpose(N15_q025_mat_new[:],(1,0))
N15_lf_uppq[:] = np.transpose(N15_q975_mat_new[:],(1,0))
N15_lf_midd[:] = np.transpose(N15_q50_mat_new[:],(1,0))

RF_file.close()











