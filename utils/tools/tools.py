import pandas as pd
import math
import os
import random
import numpy as np

def convert_to_feather(extension='csv', path='/data/input/'):
  target = ['train', 'test']
  for t in target:
    (pd.read_csv(path + t + '.' + extension, encoding="utf-8")) \
      .to_feather(path + t + '.feather')


def count_missing_values(data):
  print('column\tisnull\tcount')
  for col in data.columns:
    isnull = data[col].isnull().all()
    count = data[col].isnull().sum()
    print(col, '\t', isnull, '\t', count)


def under_sampling(data, target, perc=0.8, weight=1, random_state=None):
  # データを二つに分ける
  data_0 =  data.query(target+' == 0')
  data_1 =  data.query(target+' == 1')
  n_data_0 = len(data_0)
  n_data_1 = len(data_1)

  if (n_data_0 > n_data_1):
    under_sampled_data = pd.concat([data_0.sample(n=math.floor(n_data_1*perc*weight), random_state=random_state),
                                    data_1.sample(n=math.floor(n_data_1*perc), random_state=random_state)])
  else:
    under_sampled_data = pd.concat([data_1.sample(n=math.floor(n_data_0*perc*weight), random_state=random_state),
                                    data_0.sample(n=math.floor(n_data_0*perc), random_state=random_state)])
  return under_sampled_data.sample(frac=1).reset_index(drop=True)


def seed_everything(seed=0):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df