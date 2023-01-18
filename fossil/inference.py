import pandas as pd
import numpy as np

from .preprocessing import FossilPreprocessor, LabelEncoder
from .config import ModelsConfig
from .utils import FossilPipeline, fossil_mae


def add_missing_context(x, missing_sku):
    """
    add prodcuts from test data missing in the context dataset
    """
    missing_df = pd.DataFrame(np.nan, index=range(len(missing_sku)), columns=x.columns)
    missing_df['sku_name'] = missing_sku
    missing_df['month'] = x['month'].iloc[0]    
    missing_df['year'] = x['year'].iloc[0]     
    
    return x.append(missing_df)


def prepare_test_dates(data:pd.DataFrame):
    """Helper function to get window to be used as context for making predictions"""
    dates = sorted([(m, y) for y,m in data.groupby(['year', 'month']).groups.keys()], key=lambda d: (d[1], d[0]))

    month,yr = dates[0]
    test_dates = [(month-n,yr) if month>n else (12+month-n, yr-1) for n in range(1,ModelsConfig.LOOKBACK+1)]

    m,y = test_dates[0]
    return test_dates, [(m+n,y) if m+n<=12 else (m+n-12, y+1) for n in range(1,ModelsConfig.N_STEPS+1)]

def prepare_test_context(
        latest_data:pd.DataFrame, 
        test_data:pd.DataFrame, 
        preprocessor:FossilPreprocessor
        ):
    """
    Prepare context data and window for predictions
    """
    context = latest_data.copy()
    test_dates, _ = prepare_test_dates(test_data)

    test_sku = test_data.sku_name.unique()
    context_sku = context[context[['month', 'year']].apply(tuple, axis=1).isin(test_dates)].sku_name.unique()

    missing_sku = list(set(test_sku).difference(context_sku))
    context = context.groupby(['month','year']).apply(add_missing_context, missing_sku=missing_sku)

    return context.reset_index(drop=True), test_dates
    
def prepare_submission_data(
        test_data:pd.DataFrame, 
        primary_preds:pd.DataFrame, 
        target_cols:list, 
        pred_cols:list,
        target:str
        ):
    """
    Prepare data for meta learner as well as to be used for submssion
    """
    primary_preds = primary_preds[primary_preds.sku_name.isin(test_data.sku_name.unique())]
    pred_arr = primary_preds[target_cols].values.reshape(-1,1)

    test_cols = [c for c in primary_preds if c not in target_cols+pred_cols]

    sub_cols = [c for c in test_cols if 'target' not in c]
    sub_df = primary_preds.loc[primary_preds.index.repeat(ModelsConfig.N_STEPS)][sub_cols].reset_index(drop=True)

    sub_df[f'{target}_preds'] = pred_arr
    sub_df['time_step'] = sub_df.groupby(['sku_name','month','year']).cumcount()

    return sub_df

def make_predictions(
        test_df:pd.DataFrame, 
        test_data:pd.DataFrame, 
        feature_cols:list, 
        primary_target_cols:list, 
        primary_targets:list, 
        cv_models:list):
    """Make predictions using blended models"""
    pipeline = FossilPipeline(fossil_mae)
    test_preds = {}
    data = {}

    for i,base_model in enumerate(ModelsConfig.models):
        for target in primary_targets:
            primary_preds = pipeline.forecast(test_data, feature_cols, primary_target_cols, 
                    cv_models[f'{target}_model_{i}_{base_model}_base'], target, True, True, True)
            
            target_cols = [f'{target}_target_{i}' for i in range(ModelsConfig.N_STEPS)]
            pred_cols = [f'{target}_preds_{i}' for i in range(ModelsConfig.N_STEPS)]

            sub_df = prepare_submission_data(test_df, primary_preds, target_cols, pred_cols, target)
            _, pred_dates = prepare_test_dates(test_df) 

            sub_df['month'] = pd.DataFrame(pred_dates*int(len(sub_df)/ModelsConfig.N_STEPS)).loc[:, 0].values
            sub_df['year'] = pd.DataFrame(pred_dates*int(len(sub_df)/ModelsConfig.N_STEPS)).loc[:, 1].values
            sub_df = sub_df[sub_df['time_step']==ModelsConfig.N_STEPS-1]

            data[f'{target}_preds'] = sub_df[f'{target}_preds']
            test_preds[f'{target}_preds_{i}'] = sub_df[f'{target}_preds'].values
            sub_df.drop(columns=[f'{target}_preds'],inplace=True)            
        
        sub_df = pd.concat([sub_df, pd.DataFrame(data)], axis=1)
        
        # cols = ['preds', 'time_step', 'month', 'year', 'pred_month', 'pred_year']
        meta_features = [c for c in sub_df.columns if ('preds' in c and 'sellin' not in c)]
        meta_targets = 'Target'
        
        for j,meta_learner in  enumerate([m for m in ModelsConfig.models if m!=base_model]):
            secondary_preds = pipeline.forecast(sub_df, meta_features, meta_targets, 
                                                    cv_models[f'model_{i}_{j}_{meta_learner}'], True, False, False)
            
            test_preds[f'preds_{i}_{j}'] = secondary_preds['Target'].values  

    y_pred = np.transpose(np.concatenate([[v] for k,v in test_preds.items() if all(t not in k for t in primary_targets if t!='sellin')]), (1,0)).mean(1)
    sub_df['Target'] = y_pred

    sub_df['Item_ID'] = sub_df['sku_name'].astype(str)+'_'+sub_df['month'].astype(int).astype(str)+'_'+sub_df['year'].astype(int).astype(str)

    return sub_df[sub_df['time_step']==ModelsConfig.N_STEPS-1]