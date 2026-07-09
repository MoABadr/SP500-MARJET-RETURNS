# SP500-MARJET-RETURNS Using matlab / PYTHON
This project examines the dynamics of U.S. equity market returns Using Python/mathlab
import numpy as np
import pandas as pd
import pyflux as pf
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


# 1. LOAD AND PREPARE DATA

print('LOADING DATA...')
SSE = pd.read_csv("Shanghai SE A Share Historical Data.csv")
SPX = pd.read_csv("S&P 500 Historical Data .csv")
DXY = pd.read_csv("US_Dollar_Index_Historical_Data.csv")

def clean_price(series):
    if series.dtype == 'object':
        return series.str.replace(',', '').astype(float)
    return series.astype(float)

priceSSE = clean_price(SSE['Price'])
priceSPX = clean_price(SPX['Price'])
priceDXY = clean_price(DXY['Price'])

# Parse dates and flip (oldest to newest)
dateSSE = pd.to_datetime(SSE['Date'])[::-1].reset_index(drop=True)
dateSPX = pd.to_datetime(SPX['Date'])[::-1].reset_index(drop=True)
dateDXY = pd.to_datetime(DXY['Date'])[::-1].reset_index(drop=True)

priceSSE = priceSSE[::-1].reset_index(drop=True)
priceSPX = priceSPX[::-1].reset_index(drop=True)
priceDXY = priceDXY[::-1].reset_index(drop=True)

# Find common dates
common = pd.Index(dateSSE).intersection(pd.Index(dateSPX))
common = common.intersection(pd.Index(dateDXY))

idxSSE = dateSSE.isin(common)
idxSPX = dateSPX.isin(common)
idxDXY = dateDXY.isin(common)

# Synchronized data
prSSE = priceSSE[idxSSE].values
prSPX = priceSPX[idxSPX].values
prDXY = priceDXY[idxDXY].values
date_sync = common.values

# Calculate returns (percentage log returns)
rtSSE = 100 * np.diff(np.log(prSSE))
rtSPX = 100 * np.diff(np.log(prSPX))
rtDXY = 100 * np.diff(np.log(prDXY))
date = date_sync[1:]

print(f'Data loaded. Sample size: {len(rtSPX)} observations')


# 2. AUTOCORRELATION TESTS


print('\n' + '='*60)
print('AUTOCORRELATION TESTS')
print('='*60)

# Test for autocorrelation of returns
lb = acorr_ljungbox(rtSPX, lags=20, return_df=True)
pval = lb['lb_pvalue'].iloc[-1]
h0 = 1 if pval < 0.05 else 0
print(f'\nTEST FOR AUTOCORRELATION OF RETURNS:')
print(f'h = {h0}, p-value = {pval:.6f}')
print('Result: Reject H0 - There IS autocorrelation' if h0 else 'Result: Fail to reject H0')

# Test for autocorrelation of squared returns
lb_sq = acorr_ljungbox(rtSPX**2, lags=20, return_df=True)
pval_sq = lb_sq['lb_pvalue'].iloc[-1]
h0_sq = 1 if pval_sq < 0.05 else 0
print(f'\nTEST FOR AUTOCORRELATION OF SQUARED RETURNS:')
print(f'h = {h0_sq}, p-value = {pval_sq:.6f}')
print('Result: Reject H0 - There IS volatility clustering' if h0_sq else 'Result: Fail to reject H0')

# Ljung-Box on squared returns for multiple lags
print('\n=== LJUNG-BOX TEST ON SQUARED RETURNS ===')
print('Lag    h-statistic    p-value')
print('-'*45)
for lag in [1, 5, 10, 15, 20]:
    res = acorr_ljungbox(rtSPX**2, lags=lag, return_df=True)
    pv = res['lb_pvalue'].iloc[-1]
    hh = 1 if pv < 0.05 else 0
    print(f'{lag:<4}    {hh:<12}    {pv:.6f}')


# 3. ACF / PACF PLOTS


print('\nGENERATING ACF/PACF PLOTS...')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
plot_acf(rtSPX, ax=axes[0,0], lags=20, title='ACF of S&P 500 Returns')
plot_pacf(rtSPX, ax=axes[0,1], lags=20, title='PACF of S&P 500 Returns')
plot_acf(rtSPX**2, ax=axes[1,0], lags=20, title='ACF of Squared Returns')
plot_pacf(rtSPX**2, ax=axes[1,1], lags=20, title='PACF of Squared Returns')
plt.tight_layout()
plt.show()


# 4. ARMAX MODELS USING PYFLUX


print('\n' + '='*60)
print('ARMAX MODEL ESTIMATION (PyFlux)')
print('='*60)

# Prepare data for PyFlux
X_var_SSE = rtSSE[:-1]
X_var_DXY = rtDXY[:-1]
X_var_combined = np.column_stack([X_var_SSE, X_var_DXY])

y = rtSPX[1:]  # S&P returns (dependent variable)
exog_df = pd.DataFrame({
    'y': y,
    'x1': X_var_combined[:-1, 0],  # lagged SSE
    'x2': X_var_combined[:-1, 1]   # lagged DXY
})

# Store all models and residuals
armax_models = {}
armax_residuals = {}

# Model specifications
model_specs = [
    ('ARMAX(1,1)', 1, 1),
    ('ARMAX(1,3)', 1, 3),
    ('ARMAX(3,1)', 3, 1),
    ('ARMAX(3,3)', 3, 3),
]

for name, ar, ma in model_specs:
    print(f'\n========== {name} ==========')
    
    try:
        # Build ARIMAX model (ARMAX with exogenous variables)
        model = pf.ARIMAX(
            data=exog_df,
            formula='y ~ 1 + x1 + x2',
            ar=ar,
            ma=ma,
            family=pf.Normal()
        )
        
        # Estimate via Maximum Likelihood
        res = model.fit('MLE')
        
        # Extract information
        residuals = res.residuals.dropna().values
        n_params = len(res.params)
        
        # Calculate AIC and BIC manually
        log_likelihood = res.log_likelihood
        aic = -2 * log_likelihood + 2 * n_params
        bic = -2 * log_likelihood + n_params * np.log(len(residuals))
        
        # Ljung-Box test on residuals
        lb_test = acorr_ljungbox(residuals, lags=20, return_df=True)
        pval_resid = lb_test['lb_pvalue'].iloc[-1]
        h_resid = 1 if pval_resid < 0.05 else 0
        
        print(f'AIC = {aic:.4f}, BIC = {bic:.4f}')
        print(f'Residual Ljung-Box p-value = {pval_resid:.4f}')
        print(f'Residual Ljung-Box h = {h_resid}')
        print('\nCoefficients:')
        print(res.summary())
        
        armax_models[name] = {
            'model': model,
            'results': res,
            'aic': aic,
            'bic': bic,
            'residuals': residuals,
            'pval_resid': pval_resid
        }
        armax_residuals[name] = residuals
        
    except Exception as e:
        print(f'Error estimating {name}: {e}')

# Select best model based on AIC
best_model_name = min(armax_models.keys(), key=lambda x: armax_models[x]['aic'])
best_residuals = armax_models[best_model_name]['residuals']
print(f'\nBest model based on AIC: {best_model_name}')

# 5. CUSTOM CCC-MVGARCH IMPLEMENTATION

print('\n' + '='*60)
print('CCC-MVGARCH ESTIMATION (Custom Implementation)')
print('='*60)

class CCC_MVGARCH:
    """
    Constant Conditional Correlation Multivariate GARCH
    """
    def __init__(self, returns, p=1, q=1):
        self.returns = returns
        self.T, self.n = returns.shape
        self.p = p
        self.q = q
        
    def estimate_univariate_garch(self, series):
        """Estimate univariate GARCH for a single series"""
        # Simple GARCH(1,1) implementation using MLE
        def garch_likelihood(params, data):
            omega, alpha, beta = params
            T = len(data)
            sigma2 = np.zeros(T)
            sigma2[0] = np.var(data)
            for t in range(1, T):
                sigma2[t] = omega + alpha * data[t-1]**2 + beta * sigma2[t-1]
            ll = -0.5 * np.sum(np.log(2*np.pi*sigma2) + data**2/sigma2)
            return -ll
        
        # Initial parameters
        init_params = [0.01, 0.1, 0.8]
        bounds = [(1e-6, None), (1e-6, 1), (1e-6, 1)]
        
        result = minimize(
            garch_likelihood, 
            init_params, 
            args=(series,),
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        return result.x
    
    def estimate(self):
        """Estimate CCC-MVGARCH model"""
        # Step 1: Estimate univariate GARCH for each series
        self.omega = np.zeros(self.n)
        self.alpha = np.zeros(self.n)
        self.beta = np.zeros(self.n)
        self.volatility = np.zeros((self.T, self.n))
        self.std_residuals = np.zeros((self.T, self.n))
        
        for i in range(self.n):
            print(f'Estimating univariate GARCH for series {i+1}...')
            params = self.estimate_univariate_garch(self.returns[:, i])
            self.omega[i], self.alpha[i], self.beta[i] = params
            
            # Compute conditional variances
            sigma2 = np.zeros(self.T)
            sigma2[0] = np.var(self.returns[:, i])
            for t in range(1, self.T):
                sigma2[t] = params[0] + params[1] * self.returns[t-1, i]**2 + params[2] * sigma2[t-1]
            
            self.volatility[:, i] = np.sqrt(sigma2)
            self.std_residuals[:, i] = self.returns[:, i] / self.volatility[:, i]
        
        # Step 2: Compute constant correlation matrix
        self.correlation = np.corrcoef(self.std_residuals.T)
        
        # Step 3: Compute log-likelihood
        self.log_likelihood = self.compute_log_likelihood()
        
        return self
    
    def compute_log_likelihood(self):
        """Compute log-likelihood for CCC-MVGARCH"""
        ll = 0
        R_inv = np.linalg.inv(self.correlation)
        for t in range(self.T):
            D = np.diag(self.volatility[t, :])
            H = D @ self.correlation @ D
            ll += -0.5 * (self.n * np.log(2*np.pi) + np.log(np.linalg.det(H)) + 
                          self.returns[t, :] @ np.linalg.inv(H) @ self.returns[t, :])
        return ll
    
    def get_conditional_volatilities(self):
        return self.volatility
    
    def get_correlation_matrix(self):
        return self.correlation

# Prepare data for multivariate GARCH
# Use residuals from best ARMAX model and SSE returns
errors_matrix = np.column_stack([
    best_residuals[:len(rtSSE)-1],
    rtSSE[-len(best_residuals):-1] if len(best_residuals) <= len(rtSSE)-1 else rtSSE[:len(best_residuals)]
])

# Align lengths
min_len = min(len(errors_matrix), len(rtSSE)-1)
errors_matrix = errors_matrix[:min_len, :]
sse_aligned = rtSSE[:min_len]

# Create error matrix
error_matrix = np.column_stack([errors_matrix[:, 0], sse_aligned])

print(f'\nEstimating CCC-MVGARCH with {error_matrix.shape[0]} observations...')
ccc_model = CCC_MVGARCH(error_matrix, p=1, q=1)
ccc_results = ccc_model.estimate()

print('\nCCC-MVGARCH Results:')
print(f'Log-likelihood: {ccc_results.log_likelihood:.4f}')
print(f'Conditional Variance Parameters:')
print(f'Omega: {ccc_results.omega}')
print(f'Alpha: {ccc_results.alpha}')
print(f'Beta: {ccc_results.beta}')
print(f'\nConstant Correlation Matrix:')
print(ccc_results.correlation)

# Extract conditional volatilities
ccc_volatility = ccc_results.get_conditional_volatilities()

# Plot conditional volatilities
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
axes[0].plot(date[:len(ccc_volatility)], ccc_volatility[:, 0])
axes[0].set_title('S&P 500 Conditional Volatility (CCC-MVGARCH)')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Volatility')
axes[0].grid(True)

axes[1].plot(date[:len(ccc_volatility)], ccc_volatility[:, 1])
axes[1].set_title('SSE Conditional Volatility (CCC-MVGARCH)')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Volatility')
axes[1].grid(True)
plt.tight_layout()
plt.show()


# 6. CUSTOM DCC-MVGARCH IMPLEMENTATION


print('\n' + '='*60)
print('DCC-MVGARCH ESTIMATION (Custom Implementation)')
print('='*60)

class DCC_MVGARCH:
    """
    Dynamic Conditional Correlation Multivariate GARCH
    """
    def __init__(self, returns, p=1, q=1):
        self.returns = returns
        self.T, self.n = returns.shape
        self.p = p
        self.q = q
        
    def estimate_univariate_garch(self, series):
        """Estimate univariate GARCH for a single series"""
        def garch_likelihood(params, data):
            omega, alpha, beta = params
            T = len(data)
            sigma2 = np.zeros(T)
            sigma2[0] = np.var(data)
            for t in range(1, T):
                sigma2[t] = omega + alpha * data[t-1]**2 + beta * sigma2[t-1]
            ll = -0.5 * np.sum(np.log(2*np.pi*sigma2) + data**2/sigma2)
            return -ll
        
        init_params = [0.01, 0.1, 0.8]
        bounds = [(1e-6, None), (1e-6, 1), (1e-6, 1)]
        
        result = minimize(
            garch_likelihood, 
            init_params, 
            args=(series,),
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        return result.x
    
    def dcc_likelihood(self, params):
        """DCC log-likelihood function"""
        a, b = params
        
        # Constrain parameters
        if a < 0 or b < 0 or a + b >= 1:
            return 1e10
        
        # Compute DCC correlation dynamics
        Q_bar = np.cov(self.std_residuals.T)
        Q = np.zeros((self.T, self.n, self.n))
        R = np.zeros((self.T, self.n, self.n))
        
        Q[0] = Q_bar
        R[0] = Q_bar / np.sqrt(np.outer(np.diag(Q_bar), np.diag(Q_bar)))
        
        for t in range(1, self.T):
            z_t = self.std_residuals[t-1, :].reshape(-1, 1)
            Q[t] = (1 - a - b) * Q_bar + a * (z_t @ z_t.T) + b * Q[t-1]
            
            # Ensure positive definite
            Q[t] = (Q[t] + Q[t].T) / 2
            diag_sqrt = np.sqrt(np.diag(Q[t]))
            R[t] = Q[t] / np.outer(diag_sqrt, diag_sqrt)
            R[t] = (R[t] + R[t].T) / 2  # Ensure symmetry
        
        # Compute log-likelihood
        ll = 0
        for t in range(self.T):
            D = np.diag(self.volatility[t, :])
            try:
                H = D @ R[t] @ D
                ll += -0.5 * (self.n * np.log(2*np.pi) + np.log(np.linalg.det(H)) + 
                              self.returns[t, :] @ np.linalg.inv(H) @ self.returns[t, :])
            except:
                return 1e10
        
        self.R = R
        self.Q = Q
        return -ll
    
    def estimate(self):
        """Estimate DCC-MVGARCH model"""
        # Step 1: Estimate univariate GARCH for each series
        self.omega = np.zeros(self.n)
        self.alpha = np.zeros(self.n)
        self.beta = np.zeros(self.n)
        self.volatility = np.zeros((self.T, self.n))
        self.std_residuals = np.zeros((self.T, self.n))
        
        for i in range(self.n):
            print(f'Estimating univariate GARCH for series {i+1}...')
            params = self.estimate_univariate_garch(self.returns[:, i])
            self.omega[i], self.alpha[i], self.beta[i] = params
            
            sigma2 = np.zeros(self.T)
            sigma2[0] = np.var(self.returns[:, i])
            for t in range(1, self.T):
                sigma2[t] = params[0] + params[1] * self.returns[t-1, i]**2 + params[2] * sigma2[t-1]
            
            self.volatility[:, i] = np.sqrt(sigma2)
            self.std_residuals[:, i] = self.returns[:, i] / self.volatility[:, i]
        
        # Step 2: Estimate DCC parameters
        print('\nEstimating DCC parameters...')
        init_params = [0.05, 0.90]  # a, b
        bounds = [(0.001, 0.5), (0.001, 0.99)]
        
        result = minimize(
            self.dcc_likelihood,
            init_params,
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        self.dcc_a, self.dcc_b = result.x
        self.log_likelihood = -result.fun
        
        # Compute final correlation matrices
        self.dcc_likelihood([self.dcc_a, self.dcc_b])
        
        return self
    
    def get_conditional_volatilities(self):
        return self.volatility
    
    def get_correlations(self):
        """Return time-varying correlations"""
        if hasattr(self, 'R'):
            correlations = np.zeros((self.T, self.n, self.n))
            for t in range(self.T):
                correlations[t] = self.R[t]
            return correlations
        return None
    
    def get_rho(self):
        """Extract correlation between first two series"""
        if hasattr(self, 'R'):
            rho = np.zeros(self.T)
            for t in range(self.T):
                rho[t] = self.R[t, 0, 1]
            return rho
        return None

print(f'\nEstimating DCC-MVGARCH with {error_matrix.shape[0]} observations...')
dcc_model = DCC_MVGARCH(error_matrix, p=1, q=1)
dcc_results = dcc_model.estimate()

print('\nDCC-MVGARCH Results:')
print(f'Log-likelihood: {dcc_results.log_likelihood:.4f}')
print(f'Conditional Variance Parameters:')
print(f'Omega: {dcc_results.omega}')
print(f'Alpha: {dcc_results.alpha}')
print(f'Beta: {dcc_results.beta}')
print(f'\nDCC Parameters:')
print(f'a = {dcc_results.dcc_a:.6f}')
print(f'b = {dcc_results.dcc_b:.6f}')

# Extract dynamic correlations
rho_dcc = dcc_results.get_rho()
if rho_dcc is not None:
    print(f'\nDynamic Correlation Summary:')
    print(f'Mean: {np.mean(rho_dcc):.6f}')
    print(f'Std: {np.std(rho_dcc):.6f}')
    print(f'Min: {np.min(rho_dcc):.6f}')
    print(f'Max: {np.max(rho_dcc):.6f}')

# Extract conditional volatilities
dcc_volatility = dcc_results.get_conditional_volatilities()


# 7. PLOT DCC DYNAMIC CORRELATIONS


print('\nGENERATING DCC CORRELATION PLOTS...')

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Plot DCC dynamic correlations
if rho_dcc is not None:
    axes[0].plot(date[:len(rho_dcc)], rho_dcc, 'b-', linewidth=1.5)
    axes[0].axhline(y=np.mean(rho_dcc), color='r', linestyle='--', label=f'Mean: {np.mean(rho_dcc):.4f}')
    axes[0].set_title('Time-Varying Correlations - DCC-MVGARCH')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Correlation')
    axes[0].legend()
    axes[0].grid(True)

# Plot conditional volatilities
axes[1].plot(date[:len(dcc_volatility)], dcc_volatility[:, 0], 'g-', linewidth=1.5)
axes[1].set_title('S&P 500 Conditional Volatility - DCC-MVGARCH')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Volatility')
axes[1].grid(True)

axes[2].plot(date[:len(dcc_volatility)], dcc_volatility[:, 1], 'orange', linewidth=1.5)
axes[2].set_title('SSE Conditional Volatility - DCC-MVGARCH')
axes[2].set_xlabel('Date')
axes[2].set_ylabel('Volatility')
axes[2].grid(True)

plt.tight_layout()
plt.show()


# 8. COMPARE CCC AND DCC RESULTS


print('\n' + '='*60)
print('COMPARISON: CCC vs DCC-MVGARCH')
print('='*60)

print(f'\nCCC-MVGARCH Log-likelihood: {ccc_results.log_likelihood:.4f}')
print(f'DCC-MVGARCH Log-likelihood: {dcc_results.log_likelihood:.4f}')
print(f'\nLikelihood Ratio Test: {2 * (dcc_results.log_likelihood - ccc_results.log_likelihood):.4f}')

# CCC Correlation
ccc_corr = ccc_results.get_correlation_matrix()
print(f'\nCCC Constant Correlation: {ccc_corr[0, 1]:.6f}')

# DCC Average Correlation
if rho_dcc is not None:
    print(f'DCC Average Correlation: {np.mean(rho_dcc):.6f}')


# 9. FORECASTING WITH ARMAX
print('\n' + '='*60)
print('FORECASTING WITH ARMAX')
print('='*60)

# Use the best ARMAX model for forecasting
best_model = armax_models[best_model_name]['model']

# Prepare future exogenous variables
n_forecast = 10
future_exog = exog_df[['x1', 'x2']].iloc[-n_forecast:]

print(f'Generating {n_forecast}-step ahead forecast with {best_model_name}...')

try:
    forecast = best_model.predict(h=n_forecast, oos_data=future_exog)
    print('\nForecast:')
    print(forecast)
    
    # Plot forecast
    best_model.plot_predict(h=n_forecast, oos_data=future_exog, past_values=50)
    plt.title(f'{best_model_name} Forecast')
    plt.show()
except Exception as e:
    print(f'Forecasting error: {e}')
    print('Try reducing forecast horizon or using in-sample predictions.')

print('\n' + '='*60)
print('ANALYSIS COMPLETE')
print('='*60)


MATHLAB CODE
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


THIS CODE IS MATLAB SWITCH TO PYTHON








## Example Output
The script prints the first rows of the selected index return data.
# Running the codes with data , generating results and the results are interpreted as

The models are not statistically different from zero meaning that they hold no statistical significance. Yet, it cannot be ruled out that the true effect is zero, Certainly, there is an impact on the models that the predictive ability of the model is weak due to the insignificance of the variables p-values. The model is valid because of the auto-correlation insignificance. However, the coefficients could be jointly significant. This suggests the model's predictive power comes from the joint ARMA structure rather than any single lag, and neither Shanghai nor DXY show statistically significant predictive power for S&P 500 returns in this specification. 

For Father Analysis A multi-variative GARCH (General auto-regressive considering Heteroskedasticity) model was implemented and the results were as follows, And Constant conditional correlation was assumed. 
The GARCH equation is as follows: σt2 =ω+αε 2t−1 +βσ 2t−1Where ω= long-run variance (constant), α = ARCH effect (response to past shocks) and β = GARCH effect (persistence of volatility.  ω also represents a baseline of volatility that persists regardless of shocks, and it is significant for sp500. Also, α represents the squared past shocks that have 14.8% of current volatility. And β is explained as high persistent volatility 81.8% continue to next period. Where α + β indicate that shocks take a long time to decay as their sum is equal to 1.  the model indicates that Volatility clusters (high volatility follows high volatility), Shocks have long-lasting effects and GARCH model is appropriate for capturing volatility dynamics.  Also, the GARCH results show a half-life decay of the shock, indicating the markets' memory to restore the prices around the long run average. The time span of decay is calculated as follows ln(0.5)/ln(α + β)= ln(0.5)/ln(0.1476+0.8179) = 19.74 days. It takes around that time interval for the impact of volatility shock to decay by 50%. 
