# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
import os
import json
import datetime
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, date, time, tzinfo, timedelta
import mcclear as mc
import mcclear_angles as mca
from coordinates import coordinates
from sklearn.preprocessing import MinMaxScaler

class DataPreparationManager(object):
    orig_folder = None
    dest_folder = None
    mc_Clear_radiation_folder = None
    mc_Clear_angles_folder = None
    csv_file = None
    config_file = None
    verbose = True

    def __init__(self, orig_folder=None, dest_folder=None, mc_Clear_radiation_folder = None, mc_Clear_angles_folder = None, csv_file = None, config_file=None, verbose=True):
        
        self.orig_folder = orig_folder
        self.dest_folder = dest_folder
        self.mc_Clear_radiation_folder = mc_Clear_radiation_folder
        self.mc_Clear_angles_folder = mc_Clear_angles_folder
        self.csv_file = csv_file
        self.config_file = config_file
        self.verbose = verbose
        self.parse_config_file()
        self.check_errors()
        
    def prepare_data(self):

        file_name = self._get_file_names()        
        if os.path.isfile(os.path.join(self.dest_folder,file_name[0] + ".csv")):
            print ("Data X already exists: " + file_name[0])
            return None
        elif os.path.isfile(os.path.join(self.dest_folder, file_name[1] + ".csv")):
            print ("Data y already exists: " + file_name[1])
            return None

        csv_file = self._get_csv_file()

        if len(csv_file) == 0:
            print ("No csv file found.")
            return None
        
        data_X = self._add_coordinates(csv_file)
        data_X = self._clean_data(data_X)        
        data_X = self._solar_model(data_X)        
        data_X = self._scale_data(data_X)
        data_X = self._remove_outliers(data_X)
        data_y_persist = self._persistence_model(data_X)
        data_X,data_y = self._split_data(data_X)
        self._save_csvs(data_X, data_y, data_y_persist)        
      
    def _save_csvs(self, df_X, df_y, df_y_persist):
        """
        This function saves X and y data
        """

        # Name of the file 
        file_name = self._get_file_names()

        # if base folder doesn't exist create it
        print ('folder: '+self.dest_folder)        
        print ('Attention: Storing procedure may take a while...')
        if not os.path.isdir(self.dest_folder):
            os.makedirs(self.dest_folder)

        # destination folders for the file
        destination_folder = self.dest_folder
        

        # if destination folders doesn't exists we create them
        if not os.path.isdir(destination_folder):
            os.makedirs(destination_folder)

        # saving train data
        X_path = os.path.join(destination_folder, file_name[0] + ".csv")
        y_path = os.path.join(destination_folder, file_name[1] + ".csv")
        y_persist_path = os.path.join(destination_folder, file_name[2] + ".csv")
        df_X.to_csv(X_path, index=False)            
        df_y.to_csv(y_path, index=False)
        df_y_persist.to_csv(y_persist_path, index=False)
        
    def _get_csv_file(self):
        X_path = os.path.join(self.orig_folder, self.csv_file)
        df_X = pd.read_csv(X_path)             
        return df_X
                
    def _get_file_names(self):        
        return ["df_X","df_y","df_y_persist"]

    def parse_config_file(self):
        """This method parses the self.config file and assigns the values to attributes"""

        if self.config_file is not None:

            if self.verbose:
                print ('Parsing config file')

            with open(self.config_file) as data_file:
                config_data = json.load(data_file)
            
            if "original_folder" in config_data:
                self.orig_folder = config_data["original_folder"]

            if "destination_folder" in config_data:
                self.dest_folder = config_data["destination_folder"]

            if "mc_Clear_radiation_folder" in config_data:
                self.mc_Clear_radiation_folder = config_data["mc_Clear_radiation_folder"]

            if "mc_Clear_angles_folder" in config_data:
                self.mc_Clear_angles_folder = config_data["mc_Clear_angles_folder"]

            if "csv_file" in config_data:
                self.csv_file = config_data["csv_file"]

    def check_errors(self):
        """ This method checks if all the variables have the correct values """

        if self.dest_folder is None:
            raise Exception(7, "No destination_folder specified.")

        elif self.orig_folder is None:
            raise Exception(7, "No original_folder specified.")

        elif self.mc_Clear_radiation_folder is None:
            raise Exception(7, "No folder for the radiation data from solar model Mc Clear specified.")

        elif self.mc_Clear_angles_folder is None:
            raise Exception(7, "No folder for the solar geometry data from solar model Mc Clear specified.")

        elif self.csv_file is None:
            raise Exception(7, "No csv_file specified.")
      
    def _add_coordinates(self,csv_file):
        grouped = csv_file.groupby(['codigo']) 
        def f (group):  
            codigo = group.codigo
            lat, lng = coordinates(codigo)
            (group['latitude'],group['longitude']) = lat,lng   
            return group       
        data_X = grouped.apply(f)        
        return data_X

    def _clean_data (self,df_X):
        """ This method corrects wrong data and removes duplicates """
        df_X.loc[df_X['fecha'] == 20120406, 'fecha'] = 20150206
        df_X.loc[df_X['fecha'] == 20120408, 'fecha'] = 20150208
        df_X.loc[df_X['fecha'] == 20120511, 'fecha'] = 20150118
        df_X.loc[df_X['fecha'] == 20120526, 'fecha'] = 20150622
        df_X.loc[df_X['fecha'] == 20120607, 'fecha'] = 20150409
        df_X.loc[df_X['fecha'] == 20130429, 'fecha'] = 20150329
        df_X.loc[df_X['fecha'] == 20120714, 'fecha'] = 20150516
        df_X.loc[df_X['fecha'] == 20130502, 'fecha'] = 20150401
        df_X.loc[df_X['fecha'] == 20130525, 'fecha'] = 20150424
        df_X.loc[df_X['fecha'] == 20130612, 'fecha'] = 20150512
        df_X.loc[df_X['fecha'] == 20140129, 'fecha'] = 20150111
        df_X.loc[df_X['fecha'] == 20140624, 'fecha'] = 20150606
        df_X.loc[df_X['fecha'] == 20140624, 'fecha'] = 20150606
        df_X.loc[df_X['fecha'] == 20140626, 'fecha'] = 20150608
        df_X.loc[df_X['fecha'] == 20140627, 'fecha'] = 20150603
        df_X.loc[df_X['fecha'] == 20140628, 'fecha'] = 20150610        
        df_X['hora']=df_X[['hora']].astype(int)
        df_X['year'] = df_X['fecha'].apply(lambda x: int((str(x))[0:4]))        
        print("duplicates to be removed: ",df_X[df_X.duplicated(subset=['codigo','fecha','hora','year','radiacion'],keep=False)].shape[0])
        df_X = df_X.drop_duplicates(subset=['codigo','fecha','hora','year'],keep='first')              
        return df_X

    def _solar_model (self,df_X):
        fmt = '%Y-%m-%d %H:%M:%S %Z%z'
        df_X['day'],df_X['hour'] = df_X['fecha'].apply(lambda x:datetime.strptime(str(x),'%Y%m%d')),df_X['hora'].apply(lambda x:datetime.strptime(str(x),'%H%M').time())
        df_X['date'] = df_X.apply(lambda x: datetime.combine(x['day'],x['hour']), axis=1)
        df_X['date'] = df_X['date'].apply(lambda x: pd.Timestamp(x))
        codigos = df_X['codigo'].unique()
        # Use Mc Clear solar model
        frames = []
        print('Mc Clear: adding clear sky radiation data...')
        for codigo in codigos:
            mc_path = os.path.join(self.mc_Clear_radiation_folder,codigo+'.csv')
            m = mc.McClear(mc_path)
            df_aux = df_X[df_X['codigo']==codigo]
            print('Mc Clear: adding GHI for: '+codigo)
            df_aux['glob_radiation'] = df_aux.apply(lambda x: (m.get_irradiance(x['date']))*100, axis=1)
            df_aux['glob_radiation'] = np.where(((np.array(df_aux))[:,10]==0),np.inf,(np.array(df_aux))[:,10])
            # filter out 'radiacion'-data with wrong data types
            df_aux['radiacion'] = df_aux['radiacion'].apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna()
            print('Mc Clear: adding relativ radiation for: '+codigo)
            df_aux['rel_rad'] = df_aux.apply(lambda x: float(x['radiacion'])/(x['glob_radiation']), axis=1)
            frames.append(df_aux)
            print(codigo+" done")
        df_con = pd.concat(frames)

        print('Mc Clear: adding data for azimuth angle and elevation angle...')
        df_con['date'] = pd.to_datetime(df_con['date'])
        frames = []      
        for codigo in codigos:
            mc_path = os.path.join(self.mc_Clear_angles_folder,codigo+'.csv')
            m = mca.McClear_angles(mc_path)    
            df_aux = df_con[df_con['codigo']==codigo]
            data = m.get_solar_angles()
            df_aux = pd.merge(df_aux, data, on=["date"], how='inner')    
            frames.append(df_aux)
            print(codigo+' done')
        df_result = pd.concat(frames)
        print(len(frames))
        print(df_result.shape)
        return df_result   
    
    def _scale_data (self,df_X):
        print ("Scaling azimuth angle to values between 0 and 1...")
        mms = MinMaxScaler()
        df_X[['azimuth']] = mms.fit_transform(df_X[['azimuth']])        
        return df_X

    def _remove_outliers (self,df_X):
        print("Removing outliers with relative radiation > 1.5 (no replacement)")
        print ("Samples to be removed from whole dataset: ",(df_X[df_X["rel_rad"]>1.5]).shape[0])
        print ("At target station: ", (df_X[(df_X['codigo']=='VA01') & (df_X['rel_rad']>1.5)]).shape[0])
        df_X.loc[df_X["rel_rad"]>1.5,['radiacion','rel_rad','glob_radiation']] = np.nan
        df_X = df_X.dropna()
        return df_X

    def _persistence_model (self, df_X):
        df_y_persist = df_X[df_X['codigo']=="VA01"][['year','fecha','hora','rel_rad','glob_radiation']]
        new_labels = ['year','fecha','hora','rel_rad','glob_radiation']
        forecast_horizon = [0.5,1.0,1.5,2.0,3.0,4.0]
        target_steps = [0.5,0.5,0.5,0.5,1.0,1.0]
        df_y_persist['hora'] = df_y_persist[['hora']]//100+(df_y_persist[['hora']]%100/60)
        GHI = df_y_persist[['year','fecha','hora','glob_radiation']]
        for h in target_steps:
            GHI['hora'] = GHI[['hora']]-h
            df_y_persist = pd.merge(df_y_persist, GHI, on=["year","fecha","hora"], how='outer')
        df_y_persist = df_y_persist.fillna(0)
        for fc in forecast_horizon:
            new_labels.append("ghi_"+(str(fc))) 
        df_y_persist.columns = new_labels
        df_y_persist['p_0.5']=df_y_persist['rel_rad']*(df_y_persist['ghi_0.5']/df_y_persist['glob_radiation'])
        df_y_persist['p_1.0']=df_y_persist['rel_rad']*(df_y_persist['ghi_1.0']/df_y_persist['ghi_0.5'])
        df_y_persist['p_1.5']=df_y_persist['rel_rad']*(df_y_persist['ghi_1.5']/df_y_persist['ghi_1.0'])
        df_y_persist['p_2.0']=df_y_persist['rel_rad']*(df_y_persist['ghi_2.0']/df_y_persist['ghi_1.5'])
        df_y_persist['p_3.0']=df_y_persist['rel_rad']*(df_y_persist['ghi_3.0']/df_y_persist['ghi_2.0'])
        df_y_persist['p_4.0']=df_y_persist['rel_rad']*(df_y_persist['ghi_4.0']/df_y_persist['ghi_3.0'])
        df_y_persist = df_y_persist[['year','fecha','hora','p_0.5','p_1.0','p_1.5','p_2.0','p_3.0','p_4.0']]
        return df_y_persist
    
    def _split_data (self,df_X):
        X = df_X[['codigo','latitude','longitude','year','fecha','hora','azimuth','rel_rad']]
        y = df_X[df_X['codigo']=="VA01"][['year','fecha','hora','elevation','rel_rad']]
        df_analysis = X[['codigo','year','fecha','hora']]
        print("Overview of data set: ")
        print(df_analysis.groupby(['codigo','year']).size(),df_analysis.groupby(['year']).size())        
        return X, y
    
            
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config-file', nargs='?', action="store", dest="config_file", help="File with the parameters for the files and folders of the data.")
    parser.add_argument('--original-folder', dest="orig_folder", action="store", help="The folder where the original data is.")
    parser.add_argument('--destination-folder', dest="dest_folder", action="store", help="Folder to save the data.")
    parser.add_argument('--mc_Clear-radiation-folder', dest="mc_Clear_radiation_folder", action="store", help="The folder where the downloaded irradiation data from Mc Clear solar model is stored.")
    parser.add_argument('--mc_Clear-angles-folder', dest="mc_Clear_angles_folder", action="store", help="The folder where the downloaded solar geometry data from Mc Clear solar model is stored.")
    parser.add_argument('--csv-file', dest="csv_file", action="store", help="Name of the original csv-file.", type=str)
    
    arguments = parser.parse_args()

    config_file = arguments.config_file    
    orig_folder = arguments.orig_folder
    dest_folder = arguments.dest_folder
    mc_Clear_radiation_folder = arguments.mc_Clear_radiation_folder
    csv_file = arguments.csv_file

    datapreparationmanager = DataPreparationManager(orig_folder, dest_folder, mc_Clear_radiation_folder, csv_file, config_file)
    datapreparationmanager.prepare_data()
    
