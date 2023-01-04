# Approach
(Chambers, Mullick, & Smith, 1971) identify three forecasting techniques used to help inform decision making in the production process: qualitative techniques, time series analysis and projection, and causal models. Qualitative techniques are based on judgment and expert opinion, which are utilized to convert qualitative information, such as perceived value of a product or service, into more quantitative estimates that can be used to make more objective decisions. Time series analysis as well as causal models are both inherently quantitative techniques, making use of numeric data to inform the decision-making process.

This solution uses both time series analysis and causal modeling to identify patterns in the sales data in order to forecast demand for the various products four months into the future. The solution treats the data as a multivariate time series with several time series resulting from the various variables associated with sales of the products. The solution is designed as a stacked ensemble model with the time series analysis and projection carried out in the base model whose forecasts are then fed into the meta learner to be used as predictors for the target, which is the sell in for each product four months into the future.

# Data
Since each of the sales and inventory variables varies not only with the type of product but also with time, the data constitute multivariate time series data with multiple inputs and a multi-step output prediction task (Lee, 2018), where predictions at the final time step are the target of this challenge. The data is first processed prior to being passed into the base model to account for missing entries as well as for feature extraction and selection. This preprocessing helps structure the data to better suit the time series analysis and projection done in the base model. Forecasts from the base model are then processed and restructured before passing them to the meta-learner which tries to learn the causal relationship between the predicted values and the target.

# Stochastic Processes
A stochastic process is an indexed collection of random variables {𝑋𝑡} where 𝑋𝑡 represents measurable characteristics of interest at time t (Hillier & Lieberman, 2010), and the values, 𝑥1,𝑥2,…,𝑥𝑛 , constitute a time series. The stochastic process generating a time series can be autoregressive (Sandmann & Bober, 2010) where current or future values in the time series are a function of past values, as well as time itself i.e. 

    𝑋̂𝑡= 𝑓(𝑡,𝑋𝑡−1,𝑋𝑡−2,…,𝑋𝑡−𝑛)

This is the function that the base model attempts to approximate as it learns the patterns in the underlying stochastic processes. In the case of a linear regression, the relationship is defined as:

    𝑋̂𝑡= 𝛽1+ 𝛽2𝑡 + 𝛽3𝑋𝑡−1+ 𝛽4𝑋𝑡−2+⋯+𝛽𝑛𝑋𝑡−𝑛−2+ 𝜀𝑡

Where 𝛽𝑖’s are regression coefficients to be estimated using the data.

This model however requires several assumptions to be met in order for it to accurately represent the underlying process. Often a simple autoregressive model with only the first lagged value, AR(1), proves to be most useful in forecasting (Gujarati & Porter, 2009). Furthermore, variables related to the stochastic process being modeled can be included to improve performance of the model. The relevant assumptions as well as variables to include in the model depend on the nature of the process being modeled and are usually determined through research and experimentation. The base model therefore learns the following relationships. 

    𝑋̂𝑡= 𝑓(𝑡,𝑋𝑡−1,𝑌𝑡−1) | 𝑌̂𝑡= 𝑓(𝑡,𝑌𝑡−1)

Where 𝑌𝑡−1 represents the value of the variable related to demand(sell out) at the previous time step and 𝑌̂𝑡 is the forecast value at the current/future time step.

# Feature Selection and Engineering
Along with the date(month and year) variables and lagged values of the target, it was apparent from exploratory analysis of the data that on-hand inventory, leftover inventory, and sell out are correlated with demand. Furthermore, (Szabłowski, 2021) identifies sell out as a good predictor for sell in values. As such, the sell out variable together with the date and lagged sell in were selected as a predictors for demand in the model design. Moreover, since data from the separate channels is strongly correlated with the aggregated data, the channel variables were therefore excluded from the model.

In order to improve performance of the base model, a Fourier series was extracted from the date variables. This was done to better represent the harmonic nature of the data and ensure the model accurately captures and learns the temporal patterns within the data. Relative values of the predictors were also extracted, together with the rolling median values in order to have a better representation of the relationship between demand and the variables of interest.

![Figure 1.1 Starting Inventory and Price appear to be less correlated with demand](media/exploratory%20analysis.png)

# Causal Modeling

Unlike time series analysis and projection, the causal model tries to establish direct relationships between future demand and factors influencing it, in this case sell out. (Parthasarathy, 1994) suggests that critical factors related to demand need to be selected through analysis of past data and their effect quantified and expressed in the form of mathematical equations. These factors are then projected forward and the forecasted values are used as predictors in the causal model. This is the concept underlying the meta-leaner, where sell out forecasts at the final time step are used as predictors for the sell in variable at the corresponding time step. The model is represented as below: 

    𝑋̂̇𝑡= 𝑔(𝑌𝑡̂)

Where 𝑋̂̇𝑡 is the prediction for the demand at the final time step t, and 𝑌𝑡̂ is the sellout forecast from the base model at the corresponding time step.

# Gradient Boosting Decision Trees
Both the base model and meta learner utilize gradient boosting decision tree models to learn the patterns within the data and make forecasts. Three variants of these models(extreme gradient boost, light gradient boost, and catboost) are used since each algorithm employs a different tree building process with xgboost model growing trees depth-wise, while LightGBM trees are grown in a leaf-wise manner.

Gradient boosting decision trees are used in this solution as they offer more reliable results since they are more robust to issues such as outliers and missing data. Being ensemble models themselves, GBDT models generally give predictions with relatively lower variance. Furthermore, these algorithms do not make any assumption regarding the distribution underlying the data and are therefore more likely to better generalize.

# Data Representation
The data in this challenge comprises of multiple multivariate time series where each product makes up a single time series. (Hoseinzade & Haratizadeh, 2019) designed a general model capable of learning patterns across several financial markets and making forecasts on any of the markets. Their framework operates by extracting features from each of the markets and using these to create a single model that can generalize across all markets. This solution draws inspiration from that work, adapting the framework by carting out feature extraction from each of the products and combining them to build a single model that can be used to forecast demand on any product, including products not manufactured before.

# Group Kfold

The meta learner(causal model) depends on sell out forecasts from the base (time series) model and as such, once trained, projections must be made from data in the base model. In order to accomplish this while also minimizing data leakage, group kfold cross-validation is used to produce out-of-fold predictions. The technique is not only useful for cross-validation, but also acts as a masking component indicating to the model which time steps to use while training various iterations of the model. This allows for the model to identify different patterns in the data and increase robustness of the forecasts.

# References
Brownlee, J. (2016, December 19). How To Backtest Machine Learning Models for Time Series Forecasting. Retrieved from Machine Learning Mastery: https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/

Chambers, J. C., Mullick, S. K., & Smith, D. D. (1971, July). How to Choose the Right Forecasting Technique. Retrieved from Harvard Business Review: https://hbr.org/1971/07/how-to-choose-the-right-forecasting-technique

Gujarati, D. N., & Porter, D. C. (2009). Time Series Econometrics: Forecasting. In D. N. Gujarati, & D. C. Porter, BASIC ECONOMETRICS (pp. 773-798). New York, NY: McGraw-Hill.

Hillier, F. S., & Lieberman, G. J. (2010). Markov Chains. In F. S. Hillier, & G. J. Lieberman, Introduction to Operations Research Ninth Edition (pp. 723-725). New York, NY: McGraw-Hill.

Hoseinzade, E., & Haratizadeh, S. (2019). CNNpred: CNN-based stock market prediction using a diverse set of variables. Journal of Expert Systems with Applications.

Lee, J. B. (2018, November 12). How to Develop Convolutional Neural Network Models for Time Series Forecasting. Retrieved from Machine Learning Mastery: https://machinelearningmastery.com/how-to-develop-convolutional-neural-network-models-for-time-series-forecasting/

Parthasarathy, N. S. (1994). Demand forecasting for fertilizer marketing. Retrieved from FOOD AND AGRICULTURE ORGANIZATION OF THE UNITED NATIONS: https://www.fao.org/3/t4240e/T4240E00.htm#TOC

Sandmann, W., & Bober, O. (2010). STOCHASTIC MODELS FOR INTERMITTENT DEMANDS FORECASTING AND STOCK CONTROL. University of Bamberg, Germany.

Szabłowski, B. (2021, December 15). Sell Out Sell In Forecasting: Machine Learning for sales forecasting at Nestlé. Retrieved from Towards Data Science: https://towardsdatascience.com/sell-out-sell-in-forecasting-45637005d6ee