from ast import Return
import pickle
import pandas as pd
import numpy as np

from tqdm.notebook import tqdm
from .config import ModelsConfig

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,MinMaxScaler

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=RuntimeWarning)
tqdm.pandas()

def cyclic_encode(df: pd.DataFrame, columns: list):
    """
    """
    
    for col in columns:
        max_val = df[col].max()

        df[col + '_sin'] = np.sin(2 * np.pi * df[col] / max_val)
        df[col + '_cos'] = np.cos(2 * np.pi * df[col] / max_val)
        
    return df

def _add_fourier_series(df: pd.DataFrame):
    """
    
    
    """
    df = df.sort_values(['year','month'])
    df['12month_series'] = df['month'] % 12     
    df['9month_series'] = df['month'] % 9 
    df['6month_series'] = df['month'] % 6 
    df['quarterly_series'] = df['month'] % 3 
    df['2month_series'] = df['month'] % 2 

    return df

class LabelEncoder:
    def __init__(self, items=None, save_path=None) -> None:
        self.items = items
        self.save_path = save_path


        if self.items is not None:
            self.items = {item:ix for ix,item in enumerate(self.items)}
        
        else:
            assert self.save_path is not None, 'Please specify items to encode'
            self.items = self.load_saved_items(self.save_path)
        self.size = int(len(self.items))

    def __len__(self):
        return self.size

    def __call__(self, input):
        return self.encode(input)

    def encode(self, item):
        if item in self.items.keys():
            return int(self.items[item])  

        return self.size

    def save_items(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(self.items, f)

    def load_saved_items(self, save_path):
        with open(save_path, 'rb') as f:
            return pickle.load(f)
            
    def decode(self, ix):
        pass

class FossilData:
    def __init__(self, encoder: LabelEncoder):

        self.encoder = encoder

    def pad_sequence(self, 
                    sequence: pd.DataFrame, 
                    pad_value: int=0):
        """
        Pad products missing at a specific date

        Arguments:
        sequence  -- products at a specific date
        pad_value -- value to fill into missing products
        """
        pad_list = sorted(set(range(0, len(self.encoder) + 1)).difference(set(sequence['sku_coded'].values)))
        pad_df = pd.DataFrame(pad_value, index=range(len(pad_list)), columns=sequence.columns)
        pad_df['sku_coded'] = pad_list
        pad_df['month'] = sequence['month'].iloc[0].astype(int)
        pad_df['year'] = sequence['year'].iloc[0].astype(int)
        
        return pad_df

    def pad_sku_sequence(
                    self, 
                    features: pd.DataFrame, 
                    pad_value: int=0,
                    group_by_cols:list=['sku_name']):
        """
        Helper function to fill in missing products

        Arguments:
        features      -- DataFrame containing all features to be padded
        pad_value     -- value to pad with 
        group_by_cols -- columns to group by in order to average duplicate values
        """
        features = features.groupby(group_by_cols).mean().reset_index()

        return features.append(self.pad_sequence(features, pad_value))

    def pad_targets(self, data: pd.DataFrame, dates, pad_value: int=np.nan):
        """pad targets missing at a specific date"""
        df = data.copy()
        missing = set(dates).difference(set(df[['month', 'year']].apply(tuple, axis=1).values))
        
        if missing:
            pad_df = pd.DataFrame(pad_value, index=range(len(missing)), columns=df.columns)

            pad_df['sku_name'] = df['sku_name'].values[0]
            pad_df['month'] , pad_df['year'] = zip(*missing)

            df = df.append(pad_df)

        return df
        
    def pad_data(
            self, 
            data: pd.DataFrame,
            dates: list, 
            pad_value: float=np.nan
            ):
        """Helper function to pad missing product values"""
        sku_name = data['sku_name'].unique().tolist()
        
        df = data.copy().groupby(['month','year']).mean().reset_index()
        df['sku_name'] = sku_name*len(df)
        df = self.pad_targets(df, dates, pad_value).sort_values(by=['year','month'])
        
        return df

    def prepare_targets(self, data: pd.DataFrame, dates: list):
        df = data.copy()

        for step in range(ModelsConfig.N_STEPS):
            df[f'target_{step}'] = df.sort_values(by=['year','month'])['sellin'].shift(-(step+1))
        
        if ModelsConfig.LOOKBACK>1:
            target_cols = [f'target_{i}' for i in range(ModelsConfig.N_STEPS)]
            df['lookback_ix'] = 0
            
            for ix in range(len(dates)//ModelsConfig.LOOKBACK):
                target_dates = dates[ix*ModelsConfig.LOOKBACK:(ix+1)*ModelsConfig.LOOKBACK]
                df_ix = df[['month','year']].apply(tuple, axis=1).isin(target_dates)
                
                target_df = df.loc[df_ix, target_cols].iloc[-1].values.reshape(-1,ModelsConfig.N_STEPS)
                df.loc[df_ix, target_cols] = np.repeat(target_df, ModelsConfig.LOOKBACK, axis=0)
                
                df.loc[df_ix, 'lookback_ix'] = ix
                
        return df

    def extract_timeseries_features(self, data: pd.DataFrame, feat_cols: list, window: list=[6,9]):
        """Helper function to extract time series features"""
        df = data.copy()
        df = _add_fourier_series(df)
        df = cyclic_encode(df, [c for c in df.columns if 'series' in c]+['month'])
        
        for col in tqdm(feat_cols):
            group = df.sort_values(by=['year','month']).groupby('sku_name')[col]
            for period in window:
                
                df[f'{col}_{period}month_MM'] = group.transform(lambda x: x.rolling(period).median())
                
        return df

    def retrieve_unpadded_sequence(self, df, name, non_missing):
        return df[df[['month', 'year']].apply(tuple, axis=1).isin(non_missing[name])]

class FossilPreprocessor(FossilData):
    def __init__(
            self, 
            encoder: LabelEncoder,
            data_split:float=0.9):

        self.encoder = encoder
        self.data_split = data_split

    def extract_primary_features(
            self, 
            data: pd.DataFrame, 
            feat_cols: list, 
            dates: list, 
            return_padded: bool=False):
        """
        Extract features for the base model including fourier series 
        and rolling features such as median and mean

        Arguments:
        data          -- DataFrame from which features are extracted
        feat_cols     -- list of columns from which rolling features are to be extracted
        dates         -- list of dates for which each product was available
        return_padded -- return padded DataFrame
        """

        df = data.copy()
        non_missing = {sku:[(m, y) for y,m in df[df['sku_name']==sku].groupby(['year', 'month']).groups.keys()] for sku in tqdm(df['sku_name'].unique())}
        
        df_pad = df.groupby('sku_name').progress_apply(self.pad_data, dates=dates).reset_index(drop=True)
        df_targeted = df_pad.groupby('sku_name').progress_apply(self.prepare_targets, dates=dates).reset_index(drop=True)
        feature_df = self.extract_timeseries_features(df_targeted, feat_cols, window=[6,9])
        
        if not return_padded:
            feature_df = feature_df.groupby('sku_name').progress_apply(lambda x: self.retrieve_unpadded_sequence(x, x.name, non_missing)).reset_index(drop=True)
        
        return feature_df

    def emulate_missing(self, data: pd.DataFrame):
        """
        Helper function to emulate missing products as how 
        products would not be available in the test set. 

        Features for products randomly labeled as missing are
        replaced with nan values to be later imputed.

        The imputed features are then used to predict the actual 
        targets of the original features 
        """

        df = data.copy()
        
        non_missing = df['sku_coded'].unique()
        df = df.groupby(['month','year']).apply(self.pad_sku_sequence).reset_index(drop=True)
        
        target_cols = [f'target_{i}' for i in range(ModelsConfig.N_STEPS)]
        cols = [c for c in df.columns if c not in ['sku_coded','month','year', 'sku_name']+target_cols]
        df.loc[df['sku_coded']==len(self.encoder), cols] = np.nan
        
        return df[df['sku_coded'].isin(non_missing)]

    def extract_relative_features(self, data:pd.DataFrame):
        """Helper function to extract feature values as measured relative to their median"""
        df = data.copy()

        rel_feat_cols = [c for c in data.columns if c not in ['sku_name', 'month', 'year']
             and all(l not in c for l in ['target', 'channel'])]

        relative_cols = [f'rel_{col}' for col in rel_feat_cols]
        
        df[relative_cols] = df.groupby(['month','year'])[rel_feat_cols].apply(lambda x: x/x.median())
        return df.replace([np.inf, -np.inf], np.nan)

    def sort_dates(self, data:pd.DataFrame):
        """Helper function to sort dates in ascending order"""
        df = data.copy()

        dates_unsorted = [(m, y) for y,m in df.groupby(['year', 'month']).groups.keys()]

        return sorted(dates_unsorted, key=lambda d: (d[1], d[0]))

    def prepare_primary_data(self, data:pd.DataFrame, dates:list, inference:bool=False, store_path:str=None):
        """Helper function to prepare data for the base model"""
        df = data.copy()
        
        feature_cols = [c for c in df.columns if c not in ['sku_name', 'month', 'year']
                    and all(l not in c for l in ['target', 'channel','rel'])]

        primary_data = df[df[['month','year']].apply(tuple, axis=1).isin(dates)].copy()        
        primary_data = self.extract_primary_features(primary_data, feature_cols, dates)

        primary_data['sku_coded'] = primary_data['sku_name'].apply(self.encoder)
        primary_data = primary_data.groupby(['month','year']).progress_apply(self.emulate_missing).reset_index(drop=True)
        
        return primary_data

    def impute_missing(self, data:pd.DataFrame, inference:bool, store_path:str=None):
        df = data.copy()

        assert store_path is not None, 'save location for imputing values not specified'

        if not inference:
            cols_1 = [c for c in df.columns if c not in ['sku_name','sku_coded']]
            cols_2 = [c for c in df.columns if c not in ['month','year', 'sku_name', 'sku_coded']]
           
            sku_imputes = df.groupby('sku_name')[cols_1].median().to_dict()
            df = df.groupby('sku_name').progress_apply(lambda x: self.imputer(x, x.name, sku_imputes)).copy()
            
            date_imputes = df.groupby(['month','year'])[cols_2].median().to_dict()
            df = df.groupby(['month','year']).progress_apply(lambda x: self.imputer(x, x.name, date_imputes))

            self.save_items(f'{store_path}/sku_imputes.pkl', sku_imputes)
            self.save_items(f'{store_path}/date_imputes.pkl', date_imputes)

            return df

        sku_imputes = self.load_saved_items(f'{store_path}/sku_imputes.pkl')
        date_imputes = self.load_saved_items(f'{store_path}/date_imputes.pkl')

        df = df.groupby('sku_name').progress_apply(lambda x: self.imputer(x, x.name, sku_imputes)).copy()
        df = df.groupby(['month','year']).progress_apply(lambda x: self.imputer(x, x.name, date_imputes))

        return df

    def save_items(self, save_path, items):
        with open(save_path, 'wb') as f:
            pickle.dump(items, f)

    def load_saved_items(self, save_path):
        with open(save_path, 'rb') as f:
            return pickle.load(f)
            
    def imputer(self, x, name, impute_dict):
        for col in [c for c in x.columns if c not in ['month', 'year', 'sku_name', 'sku_coded']]:
            try:
                x[col] = x[col].fillna(impute_dict[col][name])
            except KeyError:
                continue
    
        return x

    def pca_feature_selection(self, data:pd.DataFrame, num_components: int=None, eda: bool=False):
        """
        Identify important features using PCA
        
        Arguments:
        data           -- DataFrame containing features for the model
        num_components -- number of principal components to use
        eda            -- carry out exploratory analysis to identify appropriate number of components
        """
        df = data.copy()
        features = [c for c in df.columns if all(l not in c for l in ['sku_name','sku_coded','target'])] 

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[features])

        if eda:            
            pca = PCA(random_state=ModelsConfig.SEED)
            feat_pca = pca.fit(scaled_features)        

            import matplotlib.pyplot as plt

            plt.plot(np.cumsum(pca.explained_variance_ratio_))
            plt.xlabel('number of components')
            plt.ylabel('cumulative explained variance')

            return None

        assert num_components is not None, 'number of components not specified'
        pca = PCA(num_components, random_state=ModelsConfig.SEED)
        feat_pca = pca.fit_transform(scaled_features)
        feat_importance = pd.DataFrame(pca.components_, columns = df[features].columns)

        n_pcs = pca.n_components_
        most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]

        initial_feature_names = df[features].columns
        return list(set([initial_feature_names[most_important[i]] for i in range(n_pcs)]))

    def expand_primary_data(self, data:pd.DataFrame, oof:pd.DataFrame, target_cols:list, pred_cols:list):
        """
        Expand data such that each product has observations equal to the number of time steps
        to be predicted and the targets are stacked vertically. This structure is used by the meta learner
        to learn the relationship between predictions from the base model at various time steps.

        Arguments:
        data        -- DataFrame containing features used in the base model
        oof         -- DataFrame containing OOF predictions from the base model
        target_cols -- list of target column names
        pred_cols   -- list of names assigned to columns containing predictions from the base model
        """
        secondary_data = data.copy()
        secondary_data[pred_cols] = oof

        target_arr = secondary_data[target_cols].values.reshape(-1,1)
        pred_arr = secondary_data[pred_cols].values.reshape(-1,1)

        data_cols = [c for c in secondary_data if c not in target_cols+pred_cols]
        expanded_data = secondary_data.loc[secondary_data.index.repeat(ModelsConfig.N_STEPS)][data_cols].reset_index(drop=True)

        expanded_data['target'] = target_arr
        expanded_data['preds'] = pred_arr
        expanded_data['time_step'] = expanded_data.groupby(['sku_name','month','year']).cumcount()

        return expanded_data

    def prepare_secondary_data(self, data:pd.DataFrame, oof:pd.DataFrame, target_cols:list, pred_cols:list):
        """Helper function to prepare data for meta learner"""
        expanded_data = self.expand_primary_data(data, oof, target_cols, pred_cols)

        return self.adjust_expanded_dates(expanded_data)

    def adjust_expanded_dates(self, data:pd.DataFrame):
        """
        Adjust dates to match prediction dates rather than original dates of observation
        """

        # data['month'] += ModelsConfig.N_STEPS
        # data.loc[data['month']>12, 'year'] += 1
        # data.loc[data['year']==2020, 'year'] += 1
        # data.loc[data['month']>12, 'month'] -= 12
        meta_dates_unsorted = [(m+ModelsConfig.N_STEPS, y) if m+ModelsConfig.N_STEPS<=12 
                       else (m+ModelsConfig.N_STEPS-12, y+1)
                       for y,m in data[['year','month']].apply(tuple, axis=1)]

        meta_dates = sorted(meta_dates_unsorted, key=lambda d: (d[1], d[0]))
        data[['month','year']] = pd.DataFrame(meta_dates, columns=['month','year'])
        
        return data

