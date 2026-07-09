# SP500-MARJET-RETURNS Using matlab / PYTHON
This project examines the dynamics of U.S. equity market returns Using Matlab and Python
clear; clc; close all; warning off;

%% Load data
SSE = readtable("Shanghai SE A Share Historical Data.csv");
SPX = readtable("S&P 500 Historical Data .csv");
DXY = readtable("US_Dollar_Index_Historical_Data.csv");

%% Price extraction
priceSSE = str2double(string(SSE.Price));
priceSPX = str2double(string(SPX.Price));
priceDXY = str2double(string(DXY.Price));

dateSSE = SSE.Date;
dateSPX = SPX.Date;
dateDXY = DXY.Date;

%% Flip dates (oldest to newest)
priceSSE = flipud(priceSSE);
dateSSE = flipud(dateSSE);
priceSPX = flipud(priceSPX);
dateSPX = flipud(dateSPX);
priceDXY = flipud(priceDXY);
dateDXY = flipud(dateDXY);

%% Convert to datetime
if iscell(dateSSE)
    dateSSE = datetime(dateSSE);
    dateSPX = datetime(dateSPX);
    dateDXY = datetime(dateDXY);
end

%% Convert to datenum
datenumSSE = datenum(dateSSE);
datenumSPX = datenum(dateSPX);
datenumDXY = datenum(dateDXY);

%% Find common dates
[common_spx_sse, idxSPX, idxSSE] = intersect(datenumSPX, datenumSSE);
[common_all, idxCommon, idxDXY] = intersect(common_spx_sse, datenumDXY);
idxSPX = idxSPX(idxCommon);
idxSSE = idxSSE(idxCommon);

%% Synchronized data
prSPX = priceSPX(idxSPX);
prSSE = priceSSE(idxSSE);
prDXY = priceDXY(idxDXY);
date_sync = datetime(common_all, 'ConvertFrom', 'datenum');

%% Calculate returns
rtSPX = 100 * diff(log(prSPX));
rtSSE = 100 * diff(log(prSSE));
rtDXY = 100 * diff(log(prDXY));
date = date_sync(2:end);

%% TEST FOR AUTOCORRELATION OF RETURNS
fprintf('\n========================================\n');
fprintf('TEST FOR AUTOCORRELATION OF RETURNS\n');
fprintf('========================================\n');
[h0, pval] = lbqtest(rtSPX, 'Lags', 20);
fprintf('h = %d, p-value = %.6f\n', h0, pval);
if h0 == 1
    fprintf('Result: Reject H0 - There IS autocorrelation in returns\n');
else
    fprintf('Result: Fail to reject H0 - No significant autocorrelation\n');
end

%% TEST FOR AUTOCORRELATION OF SQUARED RETURNS
fprintf('\n========================================\n');
fprintf('TEST FOR AUTOCORRELATION OF SQUARED RETURNS\n');
fprintf('========================================\n');
[h0_sq, pval_sq] = lbqtest(rtSPX.^2, 'Lags', 20);
fprintf('h = %d, p-value = %.6f\n', h0_sq, pval_sq);
if h0_sq == 1
    fprintf('Result: Reject H0 - There IS volatility clustering\n');
else
    fprintf('Result: Fail to reject H0 - No significant volatility clustering\n');
end

%% ACF AND PACF PLOTS
fprintf('\n=== GENERATING ACF AND PACF PLOTS ===\n');

figure('Name', 'ACF of S&P 500 Returns');
sacf(rtSPX, 20);
title('ACF of S&P 500 Returns');
grid on;

figure('Name', 'PACF of S&P 500 Returns');
spacf(rtSPX, 20);
title('PACF of S&P 500 Returns');
grid on;

figure('Name', 'ACF of S&P 500 Squared Returns');
sacf(rtSPX.^2, 20);
title('ACF of S&P 500 Squared Returns (Volatility Clustering)');
grid on;

figure('Name', 'PACF of S&P 500 Squared Returns');
spacf(rtSPX.^2, 20);
title('PACF of S&P 500 Squared Returns');
grid on;

%% LJUNG-BOX TEST ON SQUARED RETURNS FOR MULTIPLE LAGS
fprintf('\n=== LJUNG-BOX TEST ON SQUARED RETURNS ===\n');
fprintf('Lag    h-statistic    p-value\n');
fprintf('--------------------------------\n');
for lag = [1, 5, 10, 15, 20]
    [h_lb, pval_lb] = lbqtest(rtSPX.^2, 'Lags', lag);
    fprintf('%-4d    %-12d    %-8.6f\n', lag, h_lb, pval_lb);
end

fprintf('\n=== AUTOCORRELATION ANALYSIS COMPLETE ===\n');

%% Exogeneous Variables (Lagged returns of SSE and DXY)
X_var_SSE = rtSSE(1:end-1);
X_var_DXY = rtDXY(1:end-1);
X_var_combined = [X_var_SSE, X_var_DXY];

%% ARMAX Estimations (S&P 500 with SSE and DXY as exogenous)

% Model 1: ARMAX(1,1)
fprintf('\n========== MODEL 1: ARMAX(1,1) ==========\n');
C = 1;
p = 1:1; 
q = 1:1;
[parest_ARMA11, LL_ARMA11, err_ARMA11, se_reg_ARMA11, Diag_ARMA11,...
    VCVR_ARMA11] = armaxfilter(rtSPX(2:end),C,p,q,X_var_combined);
se1 = sqrt(diag(VCVR_ARMA11));
pval1 = 2*(1-normcdf(abs(parest_ARMA11./se1)));
[h_ARMA11, pval_ARMA11] = lbqtest(err_ARMA11, 'Lags', 20);
AIC1 = Diag_ARMA11.AIC;
BIC1 = Diag_ARMA11.SBIC;
fprintf('AIC = %.4f, BIC = %.4f, Residual p-val = %.4f\n', AIC1, BIC1, pval_ARMA11);
disp('Coefficients, Standard Errors, P-values:');
results1 = [parest_ARMA11'; se1'; pval1'];
disp(results1);

% Model 2: ARMAX(1,3)
fprintf('\n========== MODEL 2: ARMAX(1,3) ==========\n');
C = 1;
p = 1:1; 
q = 1:3;
[parest_ARMA13, LL_ARMA13, err_ARMA13, se_reg_ARMA13, Diag_ARMA13,...
    VCVR_ARMA13] = armaxfilter(rtSPX(2:end),C,p,q,X_var_combined);
se2 = sqrt(diag(VCVR_ARMA13));
pval2 = 2*(1-normcdf(abs(parest_ARMA13./se2)));
[h_ARMA13, pval_ARMA13] = lbqtest(err_ARMA13, 'Lags', 20);
AIC2 = Diag_ARMA13.AIC;
BIC2 = Diag_ARMA13.SBIC;
fprintf('AIC = %.4f, BIC = %.4f, Residual p-val = %.4f\n', AIC2, BIC2, pval_ARMA13);
disp('Coefficients, Standard Errors, P-values:');
results2 = [parest_ARMA13'; se2'; pval2'];
disp(results2);

% Model 3: ARMAX(3,1)
fprintf('\n========== MODEL 3: ARMAX(3,1) ==========\n');
C = 1;
p = 1:3; 
q = 1:1;
[parest_ARMA31, LL_ARMA31, err_ARMA31, se_reg_ARMA31, Diag_ARMA31,...
    VCVR_ARMA31] = armaxfilter(rtSPX(2:end),C,p,q,X_var_combined);
se3 = sqrt(diag(VCVR_ARMA31));
pval3 = 2*(1-normcdf(abs(parest_ARMA31./se3)));
[h_ARMA31, pval_ARMA31] = lbqtest(err_ARMA31, 'Lags', 20);
AIC3 = Diag_ARMA31.AIC;
BIC3 = Diag_ARMA31.SBIC;
fprintf('AIC = %.4f, BIC = %.4f, Residual p-val = %.4f\n', AIC3, BIC3, pval_ARMA31);
disp('Coefficients, Standard Errors, P-values:');
results3 = [parest_ARMA31'; se3'; pval3'];
disp(results3);

% Model 4: ARMAX(3,3)
fprintf('\n========== MODEL 4: ARMAX(3,3) ==========\n');
C = 1;
p = 1:3; 
q = 1:3;
[parest_ARMA33, LL_ARMA33, err_ARMA33, se_reg_ARMA33, Diag_ARMA33,...
    VCVR_ARMA33] = armaxfilter(rtSPX(2:end),C,p,q,X_var_combined);
se4 = sqrt(diag(VCVR_ARMA33));
pval4 = 2*(1-normcdf(abs(parest_ARMA33./se4)));
[h_ARMA33, pval_ARMA33] = lbqtest(err_ARMA33, 'Lags', 20);
AIC4 = Diag_ARMA33.AIC;
BIC4 = Diag_ARMA33.SBIC;
fprintf('AIC = %.4f, BIC = %.4f, Residual p-val = %.4f\n', AIC4, BIC4, pval_ARMA33);
disp('Coefficients, Standard Errors, P-values:');
results4 = [parest_ARMA33'; se4'; pval4'];
disp(results4);

%% GARCH ESTIMATION
% Define errors from your ARMAX model residuals (using ARMA33 as example)
errors = [err_ARMA33(:), rtSSE(end-length(err_ARMA33)+1:end)];

% CCC-MVGARCH
fprintf('\n========== CCC-MVGARCH ==========\n');
P = 1; % arch order
O = 0; % no asymmetry
Q = 1; % garch order
gjrtype = 1; % no asymmetry 
[parestccc, loglikeccc, Htccc, VCVccc] = ccc_mvgarch(errors, [], P, O, Q, gjrtype);
stderr = sqrt(diag(VCVccc));
for i = 1:length(errors)
    ht_ccc(i,:) = diag(Htccc(:,:,i))';
end
disp('Parameters:');
disp(parestccc');
disp('Standard Errors:');
disp(stderr');

% DCC-MVGARCH
fprintf('\n========== DCC-MVGARCH ==========\n');
[parestdcc, loglikedcc, Htdcc, VCVdcc] = dcc(errors, [], P, O, Q, gjrtype);
stderr = sqrt(diag(VCVdcc));
disp('Parameters:');
disp(parestdcc');
disp('Standard Errors:');
disp(stderr');
for i = 1:length(errors)
    ht_dcc(i,:) = diag(Htdcc(:,:,i))';
end

% Extract the dynamic correlations
T = size(Htdcc,3);
for i = 1:T
    Dt = sqrt(diag(Htdcc(:,:,i))); % extract square root of volatilities
    temp = pinv(diag(Dt)) * Htdcc(:,:,i) * pinv(diag(Dt)); % extract correlation matrix
    rho(i,1) = temp(1,2); % extract correlation
end

% Plot of dynamic correlations
figure('Name', 'Time varying correlations - DCC');
plot(date(2:length(rho)+1), rho)
title('Time varying correlations - DCC')
xlabel('Date')
ylabel('Correlation')
grid on;


Same estimations USING PYTHON
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Note: For the armaxfilter, ccc_mvgarch, and dcc functions,  specialized packages or custom implementations.

# Load data
SSE = pd.read_csv("Shanghai SE A Share Historical Data.csv")
SPX = pd.read_csv("S&P 500 Historical Data .csv")
DXY = pd.read_csv("US_Dollar_Index_Historical_Data.csv")

# Price extraction and cleaning
def clean_price_column(price_col):
    if price_col.dtype == 'object':
        # Remove commas and convert to float
        return price_col.str.replace(',', '').astype(float)
    return price_col.astype(float)

priceSSE = clean_price_column(SSE['Price'])
priceSPX = clean_price_column(SPX['Price'])
priceDXY = clean_price_column(DXY['Price'])

# Convert dates and flip (oldest to newest)
dateSSE = pd.to_datetime(SSE['Date'])[::-1].reset_index(drop=True)
dateSPX = pd.to_datetime(SPX['Date'])[::-1].reset_index(drop=True)
dateDXY = pd.to_datetime(DXY['Date'])[::-1].reset_index(drop=True)

priceSSE = priceSSE[::-1].reset_index(drop=True)
priceSPX = priceSPX[::-1].reset_index(drop=True)
priceDXY = priceDXY[::-1].reset_index(drop=True)

# Find common dates
common_dates = pd.Series(pd.Index(dateSSE).intersection(pd.Index(dateSPX)))
common_dates = pd.Series(pd.Index(common_dates).intersection(pd.Index(dateDXY)))

idxSSE = dateSSE.isin(common_dates)
idxSPX = dateSPX.isin(common_dates)
idxDXY = dateDXY.isin(common_dates)

# Synchronized data
prSPX = priceSPX[idxSPX].values
prSSE = priceSSE[idxSSE].values
prDXY = priceDXY[idxDXY].values
date_sync = common_dates.values

# Calculate returns
rtSPX = 100 * np.diff(np.log(prSPX))
rtSSE = 100 * np.diff(np.log(prSSE))
rtDXY = 100 * np.diff(np.log(prDXY))
date = date_sync[1:]

# TEST FOR AUTOCORRELATION OF RETURNS
print('\n' + '='*40)
print('TEST FOR AUTOCORRELATION OF RETURNS')
print('='*40)

# Ljung-Box test
lb_test = acorr_ljungbox(rtSPX, lags=20, return_df=True)
pval = lb_test['lb_pvalue'].values[-1]
h0 = 1 if pval < 0.05 else 0
print(f'h = {h0}, p-value = {pval:.6f}')
if h0 == 1:
    print('Result: Reject H0 - There IS autocorrelation in returns')
else:
    print('Result: Fail to reject H0 - No significant autocorrelation')

# TEST FOR AUTOCORRELATION OF SQUARED RETURNS
print('\n' + '='*40)
print('TEST FOR AUTOCORRELATION OF SQUARED RETURNS')
print('='*40)

lb_test_sq = acorr_ljungbox(rtSPX**2, lags=20, return_df=True)
pval_sq = lb_test_sq['lb_pvalue'].values[-1]
h0_sq = 1 if pval_sq < 0.05 else 0
print(f'h = {h0_sq}, p-value = {pval_sq:.6f}')
if h0_sq == 1:
    print('Result: Reject H0 - There IS volatility clustering')
else:
    print('Result: Fail to reject H0 - No significant volatility clustering')

# ACF AND PACF PLOTS
print('\n=== GENERATING ACF AND PACF PLOTS ===')

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# ACF of returns
acf_vals = acf(rtSPX, nlags=20)
axes[0, 0].stem(range(len(acf_vals)), acf_vals)
axes[0, 0].axhline(y=1.96/np.sqrt(len(rtSPX)), color='r', linestyle='--')
axes[0, 0].axhline(y=-1.96/np.sqrt(len(rtSPX)), color='r', linestyle='--')
axes[0, 0].set_title('ACF of S&P 500 Returns')
axes[0, 0].grid(True)

# PACF of returns
pacf_vals = pacf(rtSPX, nlags=20)
axes[0, 1].stem(range(len(pacf_vals)), pacf_vals)
axes[0, 1].axhline(y=1.96/np.sqrt(len(rtSPX)), color='r', linestyle='--')
axes[0, 1].axhline(y=-1.96/np.sqrt(len(rtSPX)), color='r', linestyle='--')
axes[0, 1].set_title('PACF of S&P 500 Returns')
axes[0, 1].grid(True)

# ACF of squared returns
acf_sq = acf(rtSPX**2, nlags=20)
axes[1, 0].stem(range(len(acf_sq)), acf_sq)
axes[1, 0].axhline(y=1.96/np.sqrt(len(rtSPX)), color='r', linestyle='--')
axes[1, 0].axhline(y=-1.96/np.sqrt(len(rtSPX)), color='r', linestyle='--')
axes[1, 0].set_title('ACF of S&P 500 Squared Returns')
axes[1, 0].grid(True)

# PACF of squared returns
pacf_sq = pacf(rtSPX**2, nlags=20)
axes[1, 1].stem(range(len(pacf_sq)), pacf_sq)
axes[1, 1].axhline(y=1.96/np.sqrt(len(rtSPX)), color='r', linestyle='--')
axes[1, 1].axhline(y=-1.96/np.sqrt(len(rtSPX)), color='r', linestyle='--')
axes[1, 1].set_title('PACF of S&P 500 Squared Returns')
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

# LJUNG-BOX TEST ON SQUARED RETURNS FOR MULTIPLE LAGS
print('\n=== LJUNG-BOX TEST ON SQUARED RETURNS ===')
print('Lag    h-statistic    p-value')
print('-'*40)

for lag in [1, 5, 10, 15, 20]:
    lb_test_lag = acorr_ljungbox(rtSPX**2, lags=lag, return_df=True)
    pval_lb = lb_test_lag['lb_pvalue'].values[-1]
    h_lb = 1 if pval_lb < 0.05 else 0
    print(f'{lag:<4}    {h_lb:<12}    {pval_lb:.6f}')

print('\n=== AUTOCORRELATION ANALYSIS COMPLETE ===')

# Exogeneous Variables (Lagged returns of SSE and DXY)
X_var_SSE = rtSSE[:-1]
X_var_DXY = rtDXY[:-1]
X_var_combined = np.column_stack([X_var_SSE, X_var_DXY])


# IMPORTANT NOTE: The armaxfilter function in MATLAB is not directly available  in Python. Below are alternatives using statsmodels ARIMA and custom implementations. For full armaxfilter functionality, consider using the arch package or PyFlux.


print('\n' + '='*40)
print('ARMAX MODEL ESTIMATION (ALTERNATIVE APPROACH)')
print('='*40)
print('Note: Python does not have a direct equivalent to MATLAB\'s armaxfilter.')
print('Using statsmodels ARIMA with exogenous variables as an alternative.')
print('For more advanced GARCH-MIDAS models, consider the "arch" package.')
print('='*40)

# Alternative: Use statsmodels ARIMA with exog variables
# Note: This is a simplified version - you may need to adjust for your specific needs

# Since we can't directly replicate armaxfilter, here's an alternative approach
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Fit different ARIMA models with exogenous variables
def fit_armax_exog(y, exog, ar_order, ma_order):
    """Fit ARMAX model with exogenous variables"""
    # ARIMA model with exogenous variables
    model = SARIMAX(y, 
                    order=(ar_order, 0, ma_order),
                    exog=exog,
                    trend='c',
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    result = model.fit(disp=False)
    return result

# Model 1: ARMAX(1,1)
print('\n========== MODEL 1: ARMAX(1,1) ==========')
model1 = fit_armax_exog(rtSPX[1:], X_var_combined[:-1], 1, 1)
print(f'AIC = {model1.aic:.4f}, BIC = {model1.bic:.4f}')
print('Summary:')
print(model1.summary())

# Model 2: ARMAX(1,3)
print('\n========== MODEL 2: ARMAX(1,3) ==========')
model2 = fit_armax_exog(rtSPX[1:], X_var_combined[:-1], 1, 3)
print(f'AIC = {model2.aic:.4f}, BIC = {model2.bic:.4f}')
print('Summary:')
print(model2.summary())

# Model 3: ARMAX(3,1)
print('\n========== MODEL 3: ARMAX(3,1) ==========')
model3 = fit_armax_exog(rtSPX[1:], X_var_combined[:-1], 3, 1)
print(f'AIC = {model3.aic:.4f}, BIC = {model3.bic:.4f}')
print('Summary:')
print(model3.summary())

# Model 4: ARMAX(3,3)
print('\n========== MODEL 4: ARMAX(3,3) ==========')
model4 = fit_armax_exog(rtSPX[1:], X_var_combined[:-1], 3, 3)
print(f'AIC = {model4.aic:.4f}, BIC = {model4.bic:.4f}')
print('Summary:')
print(model4.summary())


# GARCH ESTIMATION  For CCC-MVGARCH and DCC-MVGARCH, you'll need specialized packages. Below is an example using the 'arch' package for univariate GARCH. For multivariate GARCH, consider 'mgarch' or implement manually.


print('\n' + '='*40)
print('GARCH ESTIMATION')
print('='*40)
print('Note: Python has limited support for multivariate GARCH models.')
print('For univariate GARCH, use the "arch" package.')
print('For multivariate GARCH (CCC/DCC), consider implementing manually')
print('or using the "mgarch" package (if available).')
print('='*40)

# Univariate GARCH example using arch package
try:
    from arch import arch_model
    
    # Fit GARCH(1,1) model on S&P 500 returns
    am = arch_model(rtSPX, vol='Garch', p=1, q=1)
    res = am.fit(disp='off')
    print('\nUnivariate GARCH(1,1) for S&P 500:')
    print(res.summary())
    
except ImportError:
    print('\n"arch" package not installed. Install with: pip install arch')

# For multivariate GARCH, you can implement CCC-MVGARCH manually, CONSTANT CONDITIONAL CORRELATION MULATIVARIATIVE GENENAL AUTOREGRESSIVE CONDITIONAL HETEROSKEDASTICITY / AND THE DYNAMIC CONDITIONAL CORRELATION MULATIVARIATIVE GENENAL AUTOREGRESSIVE CONDITIONAL HETEROSKEDASTICITY 
print('\n' + '='*40)
print('MULTIVARIATE GARCH (CCC-MVGARCH and DCC-MVGARCH)')
print('='*40)
print('Full implementation of CCC-MVGARCH and DCC-MVGARCH requires')
print('custom coding in Python. The key steps are:')
print('1. Estimate univariate GARCH models for each series')
print('2. Extract conditional volatilities')
print('3. For CCC: Compute constant correlation matrix from standardized residuals')
print('4. For DCC: Model dynamic correlations using DCC equations')
print('='*40)


# Alternatively:

def estimate_dcc_garch_simple(returns, p=1, q=1):
    """
    Simplified DCC-GARCH estimation (illustration only)
    """
    n_assets = returns.shape[1]
    T = returns.shape[0]
    
    # Step 1: Estimate univariate GARCH for each asset
    import warnings
    warnings.filterwarnings('ignore')
    from arch import arch_model
    
    univ_vols = np.zeros((T, n_assets))
    std_resids = np.zeros((T, n_assets))
    
    for i in range(n_assets):
        am = arch_model(returns[:, i], vol='Garch', p=p, q=q)
        res = am.fit(disp='off')
        univ_vols[:, i] = res.conditional_volatility
        std_resids[:, i] = res.resid / res.conditional_volatility
    
    # Step 2: Compute unconditional correlation (simplified CCC)
    corr_matrix = np.corrcoef(std_resids.T)
    
    # Step 3: For DCC, you would estimate dynamic correlations here
    # This is a placeholder for the full DCC implementation
    
    print(f'\nCCC Correlation Matrix:')
    print(corr_matrix)
    
    return univ_vols, std_resids, corr_matrix

# (using the residuals from ARMAX model)

errors_matrix = np.column_stack([
    model4.resid,  # residuals from S&P 500 model
    rtSSE[-len(model4.resid):]  # corresponding SSE returns (simplified)
])

print('\n=== CCC-MVGARCH (Simplified) ===')
try:
    vols, std_res, corr = estimate_dcc_garch_simple(errors_matrix, p=1, q=1)
except Exception as e:
    print(f'Error in estimation: {e}')

print('\n' + '='*40)
print('END OF ANALYSIS')
print('='*40)
# Running the codes with data , generating results and the results are interpreted as

The models are not statistically different from zero meaning that they hold no statistical significance. Yet, it cannot be ruled out that the true effect is zero, Certainly, there is an impact on the models that the predictive ability of the model is weak due to the insignificance of the variables p-values. The model is valid because of the auto-correlation insignificance. However, the coefficients could be jointly significant. This suggests the model's predictive power comes from the joint ARMA structure rather than any single lag, and neither Shanghai nor DXY show statistically significant predictive power for S&P 500 returns in this specification. 

For Father Analysis A multi-variative GARCH (General auto-regressive considering Heteroskedasticity) model was implemented and the results were as follows, And Constant conditional correlation was assumed. 
The GARCH equation is as follows: σt2 =ω+αε 2t−1 +βσ 2t−1Where ω= long-run variance (constant), α = ARCH effect (response to past shocks) and β = GARCH effect (persistence of volatility.  ω also represents a baseline of volatility that persists regardless of shocks, and it is significant for sp500. Also, α represents the squared past shocks that have 14.8% of current volatility. And β is explained as high persistent volatility 81.8% continue to next period. Where α + β indicate that shocks take a long time to decay as their sum is equal to 1.  the model indicates that Volatility clusters (high volatility follows high volatility), Shocks have long-lasting effects and GARCH model is appropriate for capturing volatility dynamics.  Also, the GARCH results show a half-life decay of the shock, indicating the markets' memory to restore the prices around the long run average. The time span of decay is calculated as follows ln(0.5)/ln(α + β)= ln(0.5)/ln(0.1476+0.8179) = 19.74 days. It takes around that time interval for the impact of volatility shock to decay by 50%. 
