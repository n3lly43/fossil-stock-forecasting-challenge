import pandas as pd
import numpy as np

from .preprocessing import FossilPreprocessor, LabelEncoder
from .config import ModelsConfig
from .models.gbdt import FossilGBDT


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
    return test_dates, [(m+n,yr) if m+n<=12 else (m+n-12, yr+1) for n in range(1,ModelsConfig.N_STEPS+1)]

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
        primary_preds:np.ndarray, 
        target_cols:list, 
        pred_cols:list
        ):
    """
    Prepare data for meta learner as well as to be used for submssion
    """
    primary_preds = primary_preds[primary_preds.sku_name.isin(test_data.sku_name.unique())]
    pred_arr = primary_preds[target_cols].values.reshape(-1,1)

    test_cols = [c for c in primary_preds if c not in target_cols+pred_cols]

    sub_cols = [c for c in test_cols if 'target' not in c]
    sub_df = primary_preds.loc[primary_preds.index.repeat(ModelsConfig.N_STEPS)][sub_cols].reset_index(drop=True)

    sub_df['preds'] = pred_arr
    sub_df['time_step'] = sub_df.groupby(['sku_name','month','year']).cumcount()

    return sub_df

def make_predictions(test_df, test_data, feature_cols, target_cols, pred_cols, cv_models):
    """Make predictions using blended models"""
    gbdt_models = FossilGBDT()
    test_preds = {}

    for i,base_model in enumerate(ModelsConfig.models):
        primary_preds = gbdt_models.forecast(test_data, feature_cols, target_cols, cv_models[f'model_{i}_{base_model}_base'], True, True, True)
        
        sub_df = prepare_submission_data(test_df, primary_preds, target_cols, pred_cols)
        _, pred_dates = prepare_test_dates(test_df) 

        sub_df['month'] = pd.DataFrame(pred_dates*int(len(sub_df)/ModelsConfig.N_STEPS)).loc[:, 0].values
        sub_df['year'] = pd.DataFrame(pred_dates*int(len(sub_df)/ModelsConfig.N_STEPS)).loc[:, 1].values
        test_preds[f'preds_{i}'] = sub_df['preds'].values
        
        cols = ['preds', 'time_step', 'month', 'year', 'pred_month', 'pred_year']
        meta_features = [c for c in sub_df.columns if 'lag' in c or c in cols]
        meta_targets = 'Target'
        
        for j,meta_learner in enumerate([m for m in ModelsConfig.models if m!=base_model]):
            secondary_preds = gbdt_models.forecast(sub_df, meta_features, meta_targets, 
                                                    cv_models[f'model_{i}_{j}_{meta_learner}'], True, False, False)
            
            test_preds[f'preds_{i}_{j}'] = secondary_preds['Target'].values  

    y_pred = np.transpose(np.concatenate([[v] for v in test_preds.values()]), (1,0)).mean(1)
    sub_df['Target'] = y_pred

    sub_df['Item_ID'] = sub_df['sku_name'].astype(str)+'_'+sub_df['month'].astype(int).astype(str)+'_'+sub_df['year'].astype(int).astype(str)

    return sub_df