import pandas as pd
import numpy as np
import pickle, glob

from .utils import FossilPipeline
from .config import ModelsConfig
from .preprocessing import FossilPreprocessor

class FossilEnsemble(FossilPipeline):
    def __init__(self, criterion) -> None:
        super(FossilEnsemble, self).__init__(criterion)

    def blend_cv_models(self, 
            principal_features: list, 
            primary_targets:list,
            base_data: pd.DataFrame, 
            preprocessor: FossilPreprocessor):
        """
        Train CV models using a stacked ensemble and average predictions from all models.

        Arguments:
        principal_features -- features identified using PCA to be used in the base model
        primary_targets    -- list of features whose future values are to be forecasted in the base model
        base_data          -- data used in the base model
        preprocessor       -- fossil preprocessor used in preparing data
        """
        secondary_data, oof_preds, cv_models = self.blender(principal_features, primary_targets, base_data, preprocessor)
        # target_idx = secondary_data[secondary_data['time_step']==ModelsConfig.N_STEPS-1].index
        
        y_pred = np.transpose(np.concatenate([[v] for k,v in oof_preds.items()\
             if all(t not in k for t in primary_targets if t!='sellin')]), (1,0)).mean(1)

        y_true = secondary_data['sellin_target'].values

        blended_mae = np.absolute(np.subtract(y_true, y_pred)).mean()
        print(f'Blended MAE: {blended_mae}')
        
        return oof_preds, cv_models, blended_mae, secondary_data

    def blender(self, 
            principal_features:list, 
            primary_targets:list,
            base_data:pd.DataFrame, 
            preprocessor:FossilPreprocessor):

        oof_preds = {}
        cv_models = {}
        for i,base_model in enumerate(ModelsConfig.models):  
            secondary_data, base_preds, base_models = self.base_model_trainer(principal_features, primary_targets, 
                                                                                base_data, preprocessor, i, base_model)
            oof_preds.update(base_preds)
            cv_models.update(base_models)

            meta_preds, meta_models = self.meta_trainer(secondary_data, i, base_model)
            oof_preds.update(meta_preds)
            cv_models.update(meta_models)
        
        return secondary_data, oof_preds, cv_models

    def base_model_trainer(self, 
            principal_features:list, 
            primary_targets:list,
            base_data:pd.DataFrame, 
            preprocessor:FossilPreprocessor,
            base_model_index:int,
            base_model:str):         

        data_dict, pred_dict, model_dict = {}, {}, {}

        cols = principal_features+['month', 'year']
        feature_cols = [c for c in base_data.columns if c in cols or c in primary_targets]
        primary_target_cols = [c for c in base_data.columns if 'target' in c]

        for target in primary_targets:
            primary_cv_models = self.train_step(base_data, feature_cols, primary_target_cols, target, True, 
                                                    True, False, model_type=base_model)
            
            primary_val_mae, primary_oof = self.eval_step(base_data, feature_cols, primary_target_cols, primary_cv_models, target)
            
            target_cols = [f'{target}_target_{i}' for i in range(ModelsConfig.N_STEPS)]
            pred_cols = [f'{target}_preds_{i}' for i in range(ModelsConfig.N_STEPS)]

            secondary_data = preprocessor.prepare_secondary_data(base_data, primary_oof, target_cols, pred_cols, target)
            secondary_data = secondary_data[secondary_data['time_step']==ModelsConfig.N_STEPS-1]
            
            data_dict[f'{target}_target'] = secondary_data[f'{target}_target']
            data_dict[f'{target}_preds'] = secondary_data[f'{target}_preds']
            
            pred_dict[f'{target}_preds_{base_model_index}'] = secondary_data[secondary_data['time_step']==ModelsConfig.N_STEPS-1][f'{target}_preds']
            model_dict[f'{target}_model_{base_model_index}_{base_model}_base'] = primary_cv_models
            secondary_data.drop(columns=[f'{target}_target', f'{target}_preds'],inplace=True)
        
        return pd.concat([secondary_data, pd.DataFrame(data_dict)], axis=1), pred_dict, model_dict

    def meta_trainer(
            self, 
            secondary_data:pd.DataFrame, 
            base_model_index:int, 
            base_model:str):
        
        pred_dict, model_dict = {}, {}
        # cols = ['preds', 'time_step', 'month', 'year']
        secondary_features = [c for c in secondary_data.columns if ('preds' in c and 'sellin' not in c)]
        secondary_targets = [c for c in secondary_data.columns if 'target' in c and all(str(i) not in c for i in range(ModelsConfig.N_STEPS))]
        
        for j,meta_learner in enumerate([m for m in ModelsConfig.models if m!=base_model]):
            secondary_cv_models = self.train_step(secondary_data, secondary_features, secondary_targets, 'sellin', False, 
                                                        False, False, model_type=meta_learner,  model_level='meta')
            secondary_val_mae, secondary_oof = self.eval_step(secondary_data, secondary_features, secondary_targets,
                                        secondary_cv_models, 'sellin', False, False)

            pred_dict[f'preds_{base_model_index}_{j}'] = secondary_oof        
            model_dict[f'model_{base_model_index}_{j}_{meta_learner}'] = secondary_cv_models
            
        return pred_dict, model_dict

    def save_cv_models(self, cv_models:dict, save_path:str):
        for model_name,model in cv_models.items():
            if 'base' not in model_name:
                for i in range(ModelsConfig.FOLDS):
                    if 'lgb' in model_name:
                        model[i].save_model(f'{save_path}/{model_name}_{i}')

                    if 'xgb' in model_name:
                        pickle.dump(model[i], open(f'{save_path}/{model_name}_{i}', "wb"))

                    if 'cat' in model_name:
                        model[i].save_model(f'{save_path}/{model_name}_{i}', format="cbm")
            else:
                for i in range(ModelsConfig.N_STEPS):
                    for j in range(ModelsConfig.FOLDS):
                        if 'lgb' in model_name:
                            model[i][j].save_model(f'{save_path}/{model_name}_{i}_{j}.txt')
                        if 'xgb' in model_name:
                            pickle.dump(model[i][j], open(f'{save_path}/{model_name}_{i}_{j}.dat', "wb"))
                        if 'cat' in model_name:
                            model[i][j].save_model(f'{save_path}/{model_name}_{i}_{j}', format="cbm")

    # def load_cv_models(self, save_path):
    #     cv_models = {}
    #     models = glob.glob(f'{save_path}/*')

    #     for model_name in models:
    #         if 'lgb' in model_name:
    #             cv_models[model_name] = lgb.Booster(f'{save_path}/{model_name}')

    #         if 'xgb' in model_name:
    #             pickle.dump(model, open(f'{save_path}/{model_name}', "wb"))

    #         if 'lgb' in model_name:
    #             model.save_model(f'{save_path}/{model_name}', format="cbm")
    
