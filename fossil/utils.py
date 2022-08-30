import numpy as np

from .config import ModelsConfig
from .models.gbdt import FossilGBDT

class FossilBelnder:
    def __init__(self):
        pass

    def train_cv_models(self, principal_features, base_data, preprocessor):
        gbdt_models = FossilGBDT()

        oof_preds = {}
        cv_models = {}

        for i,base_model in enumerate(ModelsConfig.models):
            cols = principal_features+['sku_name','sku_coded', 'month', 'year']
            feature_cols = [c for c in base_data.columns if c in cols]
            target_cols = [c for c in base_data.columns if 'target' in c]

            primary_cv_models = gbdt_models.train_model(base_data, feature_cols, target_cols, True, 
                                                    True, False, model_type=base_model)
            
            primary_val_mae, primary_oof = gbdt_models.test_model(base_data, feature_cols, target_cols, primary_cv_models)
            
            target_cols = [f'target_{i}' for i in range(ModelsConfig.N_STEPS)]
            pred_cols = [f'preds_{i}' for i in range(ModelsConfig.N_STEPS)]

            secondary_data = preprocessor.prepare_secondary_data(base_data, primary_oof, target_cols, pred_cols)
            oof_preds[f'primary_{i}'] = secondary_data['preds']
            cv_models[f'primary_{i}'] = primary_cv_models
            
            for j,meta_learner in enumerate([m for m in ModelsConfig.models if m!=base_model]):
                cols = ['sku_name','sku_coded','preds', 'time_step', 'month', 'year']
                secondary_features = [c for c in secondary_data.columns if 'lag' in c or c in cols]
                secondary_targets = 'target'

                secondary_cv_models = gbdt_models.train_model(secondary_data, secondary_features, secondary_targets, False, 
                                                            False, False, model_type=meta_learner,  model_level='meta')
                secondary_val_mae, secondary_oof = gbdt_models.test_model(secondary_data, secondary_features, secondary_targets,
                                            secondary_cv_models, False, False)
                oof_preds[f'secondary_{i}_{j}'] = secondary_oof        
                cv_models[f'secondary_{i}_{j}'] = secondary_cv_models
                print('{} Base Model Validation MAE: {:.6f}\t{} Meta Learner Validation MAE: {:.6f}'.format(
                    base_model, primary_val_mae, meta_learner, secondary_val_mae))

        y_pred = np.transpose(np.concatenate([[v] for v in oof_preds.values()]), (1,0)).mean(1)
        y_true = secondary_data['target'].values
        blended_mae = np.absolute(np.subtract(y_true, y_pred)).mean()

        print(f'Blended MAE: {blended_mae}')
        
        return oof_preds, cv_models, blended_mae