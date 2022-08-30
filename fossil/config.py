class ModelsConfig():
    SEED = 1121
    N_STEPS = 4
    LOOKBACK = 1
    FOLDS = 3

    # BASE_MODEL = 'lgb'
    # META_LEARNER = 'cat'

    # checks if gpu is available
    models = ['lgb','xgb','cat']
    use_gpu = False

    lgb_params = {
                'boosting_type': 'gbdt',
                'device_type': 'gpu' if use_gpu else 'cpu',
                #'gpu_use_dp': 'true',
                'n_jobs': -1,
                'learning_rate': 0.008,
                'colsample_bytree': 0.85,
                'colsample_bynode': 0.85,
                'min_data_per_leaf': 25,
                'max_bin':63,
                'num_leaves': 125,
                'lambda_l1': 0.5,
                'lambda_l2': 0.5,
                "metric": "mae",
                "deterministic": True,
                "force_row_wise":True,
                'seed': SEED,
                'verbose':-1,
                # 'categorical_feature':[f'name:{c}' for c in categorical]
              }

    xgb_params =  dict(n_estimators=10000,
                        learning_rate=0.02, 
                        eval_metric = 'mae',
                        early_stopping_rounds=10,
                        seed=SEED)
                        
    catboost_params = dict(loss_function='MAE',
                            n_estimators=10000,
                            learning_rate=0.02, 
                            eval_metric = 'MAE',
                            random_seed=SEED)
