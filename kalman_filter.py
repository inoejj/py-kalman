import pandas as pd
from pykalman import KalmanFilter
import numpy as np
import os

def kalmanfunction(arr,level):
    measurements = np.array(arr)
    initial_state_mean = [measurements[0, 0],
                          0,
                          measurements[0, 1],
                          0]

    transition_matrix = [[1, 1, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 1],
                         [0, 0, 0, 1]]

    observation_matrix = [[1, 0, 0, 0],
                          [0, 0, 1, 0]]
    kf1 = KalmanFilter(transition_matrices=transition_matrix,
                       observation_matrices=observation_matrix,
                       initial_state_mean=initial_state_mean)

    kf1 = kf1.em(measurements, n_iter=5)

    kf2 = KalmanFilter(transition_matrices=transition_matrix,
                       observation_matrices=observation_matrix,
                       initial_state_mean=initial_state_mean,
                       observation_covariance=level * kf1.observation_covariance,
                       em_vars=['transition_covariance', 'initial_state_covariance'])

    kf2 = kf2.em(measurements, n_iter=5)
    (smoothed_state_means, smoothed_state_covariances) = kf2.smooth(measurements)

    return list(zip(smoothed_state_means[:, 0], smoothed_state_means[:, 2]))

def kalmancsv(csvfile,level=10):
    #read csv
    df = pd.read_csv(csvfile,header=None)
    df = df.iloc[1:]
    df.columns = df.iloc[0] #shift col

    # get col name
    colname = []
    a = df.iloc[0].to_list()
    b = df.iloc[1].to_list()

    for i in range(len(a)):
        colname.append(a[i]+'_'+b[i])
    #change col name
    df.columns = colname
    df = df.iloc[2:]
    df = df.reset_index(drop=True)

    #saving processed df in memory
    dforiginal = df

    #find col to drop ( probability col, or not x y)
    coltodrop = []
    for i in colname:
        if not (i.endswith('_x') or i.endswith('_y')):
            coltodrop.append(i)

    df = df.drop(coltodrop,axis=1)
    #col with only x and y
    dfkalman_col = df.columns


    #preparing loop
    c= set(a)
    c.remove('bodyparts')

    masterlist = {}
    #zip into tuples
    for i in c:
        masterlist[i] = list(zip(df[i+'_x'].astype('float32'), df[i+'_y'].astype('float32')))

    #feed tuples into kalman filter
    for i in c:
        masterlist[i]=kalmanfunction(masterlist[i],level)

    #unzip
    for i in c:
        df[i+'_x'] , df[i+'_y'] = zip(*masterlist[i])


    # ##push final df
    # finaldf = df[dfkalman_col]
    # finaldf.to_csv(os.path.join(os.path.dirname(csvfile),'output_' + os.path.basename(csvfile)),index=False)
    # print('Kalman filter applied')


    #finaldf with all cols
    finaldf = pd.concat([df[dfkalman_col],dforiginal[coltodrop]],axis=1)

    finaldf = finaldf[dforiginal.columns]

    finaldf.to_csv(os.path.join(os.path.dirname(csvfile), 'output_' + 'level_'+ str(level)+'_' +os.path.basename(csvfile)), index=False)






