from sklearn.linear_model import QuantileRegressor
import statsmodels.formula.api as smf
import pandas as pd

def VaR(data, alpha=0.05):
    """ calculate value at risk for a given alpha
    Param:
        data (pandas dataframe/series): data in interest
        alpha (float): percentile of a distribution, default alpha=0.05
        
    Return: 
        int or float: the percentile of the distibution at a givel alpha confidence interval
    """
    if isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        return data.quantile(alpha)
    else:
        raise TypeError("Expected input to be panda dataframe or series")

def CoVaR(X, Y, data, quantile=0.05, model="sklearn"):
    """ calculate conditional value at risk (CoVaR)
    Params:
        X (str): A dependent variable
        Y (str): A target variable
        data (pandas dataframe/series): data in interest
        quantile (float): percentile of a distribution
        model (str): library of choice for quantile regression (statsmodel or sklear)

    Returns:
        int or float: systemic risk value 
    """
    if model == "sklearn":
        x = data[X].values.reshape(-1,1)
        Y = data[Y]
        qr = QuantileRegressor(quantile=quantile, alpha=0, solver="highs")
        qr.fit(x, Y)
        
        lambdaCoeff = qr.coef_[0] #get the parameter coeff
    
    elif model == "statsmodel":
        variables = Y + "~" + X
        #fit quantile reg model
        quanStats = smf.quantreg(variables, data).fit(q=quantile)

        #view model summary
        lambdaCoeff = quanStats.params[X]
    else:
        raise TypeError("only 'sklearn' and 'statsmodel' are available for model parameter")

    varNormal = VaR(data, alpha=0.5)[X]
    varDistress = VaR(data, alpha = quantile)[X]

    covar = lambdaCoeff * (varNormal - varDistress) 
    return round(covar, 4)


