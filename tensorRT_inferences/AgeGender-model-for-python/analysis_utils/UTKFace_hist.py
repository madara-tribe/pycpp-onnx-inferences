import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_pd():
    gender = np.load('gender.npy')
    ages = np.load('ages.npy')
    races = np.load('races.npy')
    print(gender.shape, ages.shape, races.shape)
    df = pd.DataFrame({'age':ages, 'gender':gender, 'race':races})
    return df

def pdhist(df, name='age', save_path='age_bins.png'):
    df[name].hist(bins=20)
    plt.savefig(save_path)
    plt.show()
    
    
def main():
    df = load_pd()
    ID_RACE_MAP = {0: 'white', 1: 'black', 2: 'asian', 3: 'indian', 4: 'others'} 
    
    pdhist(df, name='age', save_path='age_bins.png')
    pdhist(df, name='gender', save_path='gender_bins.png')
    pdhist(df, name='race', save_path='race_bins.png')
    
if __name__=='__main__':
    main()
