import pandas as pd

class McClear_angles:
    def __init__(self, file):
        # Observation period;TOA;Clear sky GHI;Clear sky BHI;Clear sky DHI;Clear sky BNI
        cols = ['date','year','month','day','hour','DOY','declination','hour angle','elevation','azimuth','R']
        self.data = pd.read_csv(file, delimiter = ';', comment = '#', header=None, names = cols)
        self.data['date'] = pd.to_datetime(self.data[['day','month','year','hour']])        
                
    def get_solar_angles(self):
        result = self.data[['date','azimuth','elevation']]
        if (len(result)==0):
            print('No data found.')
        return result