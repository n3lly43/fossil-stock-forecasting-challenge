import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
import seaborn as sns
from ..config import ModelsConfig

from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
from sklearn.model_selection._split import _BaseKFold

# self defined GroupTimeSeriesSplit
class GroupTimeSeriesSplit(_BaseKFold):

    def __init__(self, n_splits=5, *, max_train_size=None):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_size = max_train_size

    def split(self, X, y=None, indices=None):


        n_splits = self.n_splits
        n_folds = n_splits + 1
        
        # X, y = indexable(X, y)
        # n_samples = _num_samples(X)
        
        # indices = np.arange(n_samples)
        # group_counts = np.unique(groups, return_counts=True)[1]
        
        # groups = np.split(indices, np.cumsum(group_counts)[:-1])
        # groups = [X[X[['month', 'year']].apply(tuple, axis=1).isin(date)].index for date in dates]
        # print(np.cumsum(group_counts))
        # n_groups = _num_samples(groups)
        
        if n_folds > len(indices):
            raise ValueError(
                ("Cannot have number of folds ={0} greater"
                 " than available dates: {1}.").format(n_folds, len(dates)))
                 
        test_size = (len(indices) // n_folds)
        test_starts = range(test_size + len(indices) % n_folds,
                            len(indices), test_size)
        for test_start in test_starts:
            # print(groups)
            if self.max_train_size:
                train_start = np.searchsorted(
                    np.cumsum(
                        group_counts[:test_start][::-1])[::-1] < self.max_train_size + 1, 
                        True)
                yield (np.concatenate(groups[train_start:test_start]),
                       np.concatenate(groups[test_start:test_start + test_size]))
            else:
                yield (np.concatenate(indices[:test_start]), indices[test_start])

def update_feature_importances(x_train, model, fold, step, feature_importance):
    # feature importance
    _importance = pd.DataFrame()
    _importance["feature"] = x_train.columns
    _importance[f"Step_{step+1}_fold_{fold+1}_importance"] = model.feature_importance()
    # fold_importance["fold"] = fold + 1
    
    return pd.concat([feature_importance, _importance], axis=1)

def cv_feature_importance_plot(data: pd.DataFrame, 
                               feature_cols: list,
                               target_cols: list,
                               folds: int, 
                               cv_models: list,
                               recursive: bool=False,
                               plot_feature_importance: bool=True):

    feature_importance = pd.DataFrame()
    
    print('\n')
    kf = GroupKFold(folds)
    oof = np.zeros((len(data),ModelsConfig.N_STEPS))

    features = data[feature_cols].copy()
    targets = data[target_cols].copy()
    
    group = features['month'].astype(str)+'_'+features['year'].astype(str)

    for i,model in enumerate(cv_models):
        for fold, (train_id, val_id) in enumerate(kf.split(features, targets, group)):
            feature_importance = update_feature_importances(features.iloc[train_id].drop(columns=['sku_name', 'sku_coded']), model[fold], fold, i, feature_importance)
            
            oof[val_id, i] = model[fold].predict(features.iloc[val_id].drop(columns=['sku_name','sku_coded']), start_iteration=-1)
                
            if plot_feature_importance:
                plot_Imp_lgb(model[fold], features.iloc[train_id].drop(columns=['sku_name', 'sku_coded']), f'step_{i+1}_fold_{fold+1}')
            
        if recursive:
            features.loc[:, f'preds_{i}'] = oof[:, i]
                
    return feature_importance

def plot_Imp_lgb(model, X, score, save_path=None):
    '''
    A funtion to make feature importance plot for LGBM.
    Returns feature-impotance plot.

    args: model -trained model
    args: X     - features used to train model
    
    '''
    feature_imp = pd.DataFrame({'Value':model.feature_importance(),'Feature':X.columns})
    plt.figure(figsize=(40, 20))
    sns.set(font_scale = 1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", 
                                                        ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(f'{save_path}/lgbm_importances{score}.png')
    plt.show()