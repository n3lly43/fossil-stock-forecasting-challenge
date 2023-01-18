import gc
import pandas as pd
import numpy as np

import lightgbm as lgb
import xgboost as xgb
import statsmodels.api as sm
from catboost import CatBoostRegressor
from .config import ModelsConfig

from time import time
from sklearn.model_selection import GroupKFold

from .config import ModelsConfig

def fossil_mae(y_true, y_pred):
    '''MAE calculator'''
    return np.absolute(np.subtract(y_true, y_pred))

class FossilPipeline:
    def __init__(self, criterion) -> None:
        self.criterion = criterion
    
    def prepare_cv_model(
            self, 
            xy_train: tuple, 
            xy_val: tuple, 
            model_type: str):
        """
        Train cross validation models.

        Arguments:
        xy_train   -- tuple of features and targets from training data
        xy_val     -- tuple of features and targets from validation set
        model_type -- string denoting type of gbdt model to implement
        """

        if model_type=='lgb':
            train_set = lgb.Dataset(*xy_train)
            val_set = lgb.Dataset(*xy_val)

            model = lgb.train(ModelsConfig.lgb_params, 
                            train_set,
                            num_boost_round = 100000,
                            callbacks=[
                                lgb.early_stopping(stopping_rounds=5),
                                lgb.log_evaluation(period=500)],
                            valid_sets = [train_set, val_set], 
                            )
    
        X_train, y_train = xy_train
    
        if model_type=='cat':
            model = CatBoostRegressor(verbose=500, use_best_model=True, **ModelsConfig.catboost_params)

            model.fit(X_train, y_train,
                    eval_set=[xy_train, xy_val],
                    early_stopping_rounds=10,
                    verbose=500)
            
        if model_type=='xgb':
            model = xgb.XGBRegressor(**ModelsConfig.xgb_params)

            model.fit(X_train, y_train,
                    eval_set=[xy_train, xy_val],
                    verbose=500)

        if model_type=='reg':
            ols_model = sm.GLS(y_train, X_train)
            model = ols_model.fit()

        return model

    def prepare_kfold_data(
            self, 
            features: pd.DataFrame, 
            targets: pd.DataFrame,
            target_name:str,
            train_id: pd.DataFrame.index,
            val_id: pd.DataFrame.index,
            model_level: str,
            step: int=None
            ):
        """
        KFold training of CV models.

        Arguments:
        features    -- dataframe containing feature columns
        targets     -- targets corresponding to the features
        target_name -- name of target if training base model
        train_id    -- indices of training data for kth fold
        val_id      -- indices of validation data for kth fold
        model_level -- level of model in the stack ensemble, i.e. base or meta model
        step        -- prediction time step
        """
        x_train = features.iloc[train_id]
        x_val = features.iloc[val_id]

        if model_level=='base':
            assert step is not None, 'prediction time step not specified'

            y_train = targets.iloc[train_id][f'{target_name}_target_{step}']
            y_val = targets.iloc[val_id][f'{target_name}_target_{step}']
        
        else:
            y_train = targets.iloc[train_id]
            y_val = targets.iloc[val_id]

        return (x_train,y_train), (x_val,y_val)    

    def train_step(
            self,
            data: pd.DataFrame,
            feature_cols: list,
            target_cols: list,
            target:str=None,
            multi_step: bool=True,
            recursive_train: bool=False,
            store_cv_models: bool=False,
            model_type: str='lgb',
            model_level: str='base'
            ):
        """
        Train gbdt models using kfold cross validation.

        Arguments:
        data            -- Training dataframe
        feature_cols    -- List of feature columns
        target_cols     -- List of target columns
        target          -- Name of column used as target in base model
        multi_step      -- whether or not predictions are made on multiple time steps
        recursive_train -- whether or not predictions are used recursively as features
        store_cv_models -- whether or not to save models to file on disk
        model_type      -- type of gbdt model to use i.e. LightGBM, XGBoost, CatBoost, etc
        model_level     -- level of model in ensemble stack i.e. base model, meta-learner
        """
        target_cols = [c for c in target_cols if target in c]

        features = data[feature_cols].copy()
        targets = data[target_cols].copy()
        
        group = data['month'].astype(str)+'_'+data['year'].astype(str)

        cv_models = []
        start_time = time()
        kfold = GroupKFold(ModelsConfig.FOLDS)
        
        if multi_step:
            oof = np.zeros((len(data),ModelsConfig.N_STEPS))
            for step in range(ModelsConfig.N_STEPS):
                print(f'Training {model_type} model for timestep {step+1} {target} forecasting')
                models = []
                
                for fold, (train_id, val_id) in enumerate(kfold.split(features, targets, group)):
                    print('\n')
                    print(f'Training fold {fold+1}')

                    xy_train, xy_val = self.prepare_kfold_data(features, targets, target, train_id, val_id, 
                                                                model_level, step)

                    model = self.prepare_cv_model(xy_train, xy_val, model_type)
                    oof_preds = model.predict(xy_val[0])   

                    if store_cv_models:
                        model.save_model(f'base_cv_model_step{step+1}_{fold+1}')

                    oof[val_id, step] = oof_preds
                    models.append(model) 
    
                if recursive_train:
                    features.loc[:, f'{target}_preds_{step}'] = oof[:, step]

                gc.collect()

                print("Elapsed {:.2f} mins".format((time() - start_time)/60))
                print('-'*50)
                print('\n')

                gc.collect()
                cv_models.append(models)

            return cv_models
            
        for fold, (train_id, val_id) in enumerate(kfold.split(features, targets, group)):
            print('\n')
            print(f'Training {model_type} model fold {fold+1}')
            
            xy_train, xy_val = self.prepare_kfold_data(features, targets, 'sellin', train_id, val_id, 
                                                        model_level, model_type)
            model = self.prepare_cv_model(xy_train, xy_val, model_type)
            
            if store_cv_models:
                model.save_model(f'meta_cv_model_{fold+1}')

            gc.collect()

            print("Elapsed {:.2f} mins".format((time() - start_time)/60))
            print('-'*50)
            print('\n')

            gc.collect()
            cv_models.append(model)

        return cv_models

    def eval_step(
            self,
            data: pd.DataFrame,
            feature_cols: list,
            target_cols: list,
            cv_models: list,
            target:str='sellin',
            multi_step: bool=True,
            recursive_train: bool=True
            ):
        """
        Test model performance using OOF predictions
        """
        target_cols = [c for c in target_cols if target in c]

        features = data[feature_cols].copy()
        targets = data[target_cols].copy()
        
        group = data['month'].astype(str)+'_'+data['year'].astype(str)

        start_time = time()   
        kfold = GroupKFold(ModelsConfig.FOLDS)

        if multi_step:
            oof = np.zeros((len(data),ModelsConfig.N_STEPS))
            for step in range(ModelsConfig.N_STEPS):
                print(f"Making timestep {step+1} predictions")
                for fold, (train_id, val_id) in enumerate(kfold.split(features, targets, group)):
                    x_val = features.iloc[val_id]
                   
                    oof[val_id, step] = cv_models[step][fold].predict(x_val)
            
                if recursive_train:
                    features.loc[:, f'{target}_preds_{step}'] = oof[:, step]
                    
                step_mae = self.criterion(data[f'{target}_target_{step}'], features[f'{target}_preds_{step}']).mean()
                print(f'Val MAE: {step_mae}')
                print('\n')
            val_mae = self.evaluate(data, oof, target)

        else:
            oof = np.zeros(len(data)) 
            # target_idx = features[features['time_step']==ModelsConfig.N_STEPS-1].index
            
            for fold, (train_id, val_id) in enumerate(kfold.split(features, targets, group)):
                print('\n')
                print(f'Making fold {fold+1} predictions')
                
                x_val = features.iloc[val_id]
                y_val = targets.iloc[val_id]
                
                oof[val_id] = cv_models[fold].predict(x_val)
                # print(oof[val_id].shape, y_val.shape, cv_models[fold].predict(x_val).shape)
                fold_mae = self.criterion(y_val.values, oof[val_id].reshape(-1,1)).mean()
                print(f'Val MAE: {fold_mae}')
                print('\n')
            # oof = oof[target_idx]
            val_mae = self.criterion(targets.values, oof.reshape(-1,1)).mean()

        print(f'Average Val MAE: {val_mae}')
        print('-'*50)
        print('\n')
        
        return val_mae, oof

    def evaluate(self, data, oof, target): 
        """Helper function to evaluate OOF predictions using desired metric"""
        df = data.copy()
        oof_cols = [f'{target}_preds_{i}' for i in range(ModelsConfig.N_STEPS)]
        target_cols = [c for c in df.columns if 'target' in c and target in c]
        
        df[oof_cols] = oof
        if ModelsConfig.LOOKBACK>1:
            df = df.groupby(['sku_name','lookback_ix']).mean().reset_index()
            
        pred_df = pd.DataFrame(np.repeat(df['sku_name'], ModelsConfig.N_STEPS), columns=['sku_name']).reset_index(drop=True)
        
        pred_df['oof'] = df[oof_cols].values.reshape(-1,1)
        pred_df['target'] = df[target_cols].values.reshape(-1,1)
        
        oof_mae = self.criterion(pred_df['target'], pred_df['oof']).mean()
        
        return oof_mae

    def forecast(
            self, 
            data: pd.DataFrame,
            feature_cols: list,
            target_cols: list,
            models: list,
            target:str='sellin',
            cv: bool=True,
            multi_step: bool=True,
            recursive: bool=True
        ):
        """
        Make predictions on unseen data
        """
        features = data[feature_cols].copy()
        data[target_cols]=0

        if cv:
            if multi_step:
                for step in range(ModelsConfig.N_STEPS):
                    for fold in range(ModelsConfig.FOLDS):
                        data[f'{target}_target_{step}'] += models[step][fold].predict(features)/ModelsConfig.FOLDS
                    if recursive:
                        features[f'{target}_preds_{step}'] = data[f'{target}_target_{step}']

            else:
                for fold in range(ModelsConfig.FOLDS):
                    data['Target'] += models[fold].predict(features)/ModelsConfig.FOLDS

        return data
