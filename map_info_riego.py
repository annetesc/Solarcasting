import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from math import sin, cos, sqrt, atan2, radians


def plot_map_stations():
    data_X = pd.read_csv('C:/Users/Anni/CS/_Pro/TFG_AE/Inforiego_map/df_stations.csv')
    def man_dist_2_target(lat,lon):
        R = 6373.0
        lat_station = radians(lat)
        lon_station = radians(lon)        
        lat_target = radians(42.166240)
        lon_target = radians(-5.264526)
        dlon = lon_target - lon_station
        dlat = lat_target - lat_station     

        a1 = sin(0 / 2)**2 + cos(lat_target) * cos(lat_target) * sin(dlon / 2)**2
        a2 = sin(dlat / 2)**2 + cos(lat_station) * cos(lat_target) * sin(0 / 2)**2
        c1 = 2 * atan2(sqrt(a1), sqrt(1 - a1))
        c2 = 2 * atan2(sqrt(a2), sqrt(1 - a2))
        WO_distance = R * c1*np.sign(lon_station - lon_target)
        NS_distance = R * c2*np.sign(lat_station - lat_target)
        return [WO_distance,NS_distance]

    df_stations = data_X[['codigo','latitude','longitude']]
    df_stations.drop_duplicates(inplace=True)
    df_stations.reset_index(drop=True,inplace=True)
    df_stations['WO_dist'],df_stations['NS_dist'] = df_stations.apply(lambda x: man_dist_2_target(x['latitude'],x['longitude'])[0], axis=1),df_stations.apply(lambda x: man_dist_2_target(x['latitude'],x['longitude'])[1], axis=1)
    castilla_and_leon_img=mpimg.imread('C:/Users/Anni/CS/_Pro/TFG_AE/Inforiego_map/Castilla_and_Leon_Provinces.png')
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(12,9))
    plt.title("InfoRiego stations",fontsize=20)
    plt.xlabel('X [km]',fontsize=15)
    plt.ylabel('Y [km]',fontsize=15)
    ax = fig.add_subplot(111)
    ax.imshow(castilla_and_leon_img, extent=[-140, 290,-240, 118], alpha=0.3)
    data = df_stations.values
    plt.scatter(data[:, 3], data[:, 4], marker='o')

    for label, x, y in zip(data[:, 0], data[:, 3], data[:, 4]):
        ax.annotate(
        label,
        xy=(x, y), xytext=(0,2),
        textcoords='offset points', ha='right', va='bottom')
        
    plt.show()

plot_map_stations()




