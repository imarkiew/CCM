import pandas as pd
from pandas import Series
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import ccf
from statsmodels.tsa.arima_model import ARMA
import seaborn as sns
import numpy as np
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from scipy.signal import spectrogram


# path to data
PATH_TO_DATA = './data/PJMW_hourly.csv'
# path to folder where we want to save plots
PATH_TO_PLOTS = './plots'
# type of separator in data file
SEPARATOR = ','
# header location
HEADER = 0
# take data from the last 8 years
TAKE_LAST = '8Y'
# divide into a training and test set
SPLIT_DATE = '2016-06-01' # '2018-06-01'
# the maximum order of the model
MAX_P = 16
# number of phases to plot
NR_OF_PHASES_TO_PLOT = 5
# other model parameters
PARAMS = {'method': 'mle', 'solver': 'lbfgs',  'maxiter': 500, 'trend': 'c', 'transparams': True, 'disp': False}

# the function returns a list of dictionaries
def get_k_roots_with_largest_radiuses(coeffs, k):
    roots = np.roots(coeffs)
    are_original_radiuses_in_unit_circle = []
    radiuses = []
    phases = []
    for original_root in roots:
        original_radius = np.abs(original_root)
        if original_radius > 1:
            are_original_radiuses_in_unit_circle.append(False)
            inversed_root = complex_inverse(original_root)
            radiuses.append(np.abs(inversed_root))
            phases.append(convert_phase_to_new_domain(np.angle(inversed_root, deg=True)))
        else:
            are_original_radiuses_in_unit_circle.append(True)
            radiuses.append(original_radius)
            phases.append(convert_phase_to_new_domain(np.angle(original_root, deg=True)))

    roots = sorted([{'was_original_radius_in_unit_circle': was_original_radius_in_unit_circle,
                     'radius': radius,
                     'phase': phase} for was_original_radius_in_unit_circle, radius, phase in zip(are_original_radiuses_in_unit_circle, radiuses, phases)],
                   key=lambda key: key['radius'], reverse=True)
    return roots[0:k]

def complex_inverse(z):
    return 1.0 / np.conjugate(z)

def convert_phase_to_new_domain(phase):
    if phase >= 180:
        return phase - 360
    else:
        return phase

# reading the data
series = Series.from_csv(PATH_TO_DATA, sep=SEPARATOR, header=HEADER)

# changing raw index to DatetimeIndex
series.index = pd.to_datetime(series.index, infer_datetime_format='True')

# sorting
series.sort_index(inplace=True)

# take measurements for a recent period, group values by month in each year and then calculate mean
series_monthly = series.last(TAKE_LAST).groupby(pd.Grouper(freq='M')).mean()

# plot a whole time series
f = plt.figure()
plt.plot(series_monthly)
plt.gcf().set_size_inches(10, plt.gcf().get_size_inches()[1])
plt.title('Średnie zużycie energii elektrycznej w skali miesięcznej na przestrzeni 8 lat')
plt.xlabel('Data')
plt.ylabel('Zużycie [MW]')
plt.grid()
f.savefig(PATH_TO_PLOTS + '/timeSeries.pdf', bbox_inches='tight')
plt.show()

# plot autocorrelation
f = plot_acf(series_monthly, unbiased=True, alpha=0.05)
plt.title('Funkcja autokorelacja')
plt.xlabel('Opóźnienie')
plt.ylabel('Autokorelacja')
plt.gcf().set_size_inches(10, plt.gcf().get_size_inches()[1])
f.savefig(PATH_TO_PLOTS + '/autocorrelation.pdf', bbox_inches='tight')
plt.show()

f = plt.figure()
freq, t1, Sxx = spectrogram(series_monthly.values, fs=3.86e-7, nfft=128,  window=('tukey', 0.25),
                            nperseg=6, noverlap=4, detrend=False, scaling='density')
plt.pcolormesh(t1, freq, Sxx)
plt.xlabel('Czas [s]')
plt.ylabel('Częstotliwość [Hz]')
plt.colorbar().set_label('Widmowa gęstość mocy [V^2/Hz]')
f.savefig(PATH_TO_PLOTS + '/spectrogram.pdf', bbox_inches='tight')
plt.show()

# signal decomposition = trend + seasonal + error
decomposition = seasonal_decompose(series_monthly, model="additive")
f = decomposition.plot()
f.savefig(PATH_TO_PLOTS + '/decomposition.pdf', bbox_inches='tight')
plt.show()

# split train-test
train = series_monthly.loc[series_monthly.index < SPLIT_DATE]
test = series_monthly.loc[series_monthly.index >= SPLIT_DATE]
print('Train size = {} %'.format(100*len(train)/len(series_monthly)))
print('Test size = {} %'.format(100*len(test)/len(series_monthly)))
f = plt.figure()
plt.plot(train)
plt.plot(test)
plt.gcf().set_size_inches(10, plt.gcf().get_size_inches()[1])
plt.title('Podział na zbiór trenujący i testowy')
plt.xlabel('Data')
plt.ylabel('Zużycie [MW]')
plt.legend(['train', 'test'])
plt.grid()
f.savefig(PATH_TO_PLOTS + '/timeSeriesTrainTest.pdf', bbox_inches='tight')
plt.show()

# check different lags
aics = {}
for p in range(1, MAX_P + 1):
    print('Model lag = {} from {}'.format(p, MAX_P))
    model = ARMA(train, order=(p, 0)).fit(**PARAMS)
    aics[p] = model.aic

# plot aic(lag)
f = plt.figure()
plt.plot(aics.keys(), aics.values(), 'bo')
plt.title('AIC(Lag)')
plt.xlabel('Lag')
plt.ylabel('AIC')
plt.xticks(range(1, len(aics) + 1))
plt.grid()
f.savefig(PATH_TO_PLOTS + '/aicLag.pdf', bbox_inches='tight')
plt.show()

# find the lag with the smallest aic
lag = min(aics, key=aics.get)
print('Optimal lag = {}'.format(lag))
# lag = 13

# train-test procedure using moving window
series_len = len(series_monthly)
train_len = len(train)
test_len = len(test)
y_pred = pd.Series([])
coefficients = []
confidence_intervals = [[], []]
for i in range(test_len):
    print('Train - test iteration: i = {} from {}'.format(i + 1, test_len))
    dynamic_train = series_monthly.iloc[i:i + train_len]
    model = ARMA(dynamic_train, order=(lag, 0)).fit(**PARAMS)
    results = model.forecast(1)
    confidence_intervals[0].extend([results[2][0][0]])
    confidence_intervals[1].extend([results[2][0][1]])
    y_pred = y_pred.append(pd.Series(results[0], index=[test.index[i]]), verify_integrity=True)
    coefficients.append(model.params)

# plot test-predicted data
f = plt.figure()
plt.plot(test, color='blue')
plt.plot(y_pred, color='orange')
plt.fill_between(test.index, confidence_intervals[0], confidence_intervals[1], color='lightgrey')
plt.gcf().set_size_inches(10, plt.gcf().get_size_inches()[1])
plt.title('Model AR')
plt.xlabel('Data')
plt.ylabel('Zużycie [MW]')
plt.legend(['test', 'AR({})'.format(lag)])
plt.grid()
f.savefig(PATH_TO_PLOTS + '/timeSeriesPredTest.pdf', bbox_inches='tight')
plt.show()

# plot crosscorrelation
f = plt.figure()
plt.plot(ccf(test, y_pred, unbiased=True))
plt.title('Korelacja krzyżowa szeregów test i pred')
plt.xlabel('Opóźnienie')
plt.ylabel('Korelacja')
plt.grid()
f.savefig(PATH_TO_PLOTS + '/crosscorrelation.pdf', bbox_inches='tight')
plt.show()

# plot residues
res = y_pred - test
f = plt.figure()
plt.plot(res)
plt.gcf().set_size_inches(10, plt.gcf().get_size_inches()[1])
plt.title('Residua dla modelu AR({})'.format(lag))
plt.xlabel('Data')
plt.ylabel('Residu*a')
plt.grid()
f.savefig(PATH_TO_PLOTS + '/residuum.pdf', bbox_inches='tight')
plt.show()

# plot box-plot for residues
f = plt.figure()
ax = sns.boxplot(y=res, palette='Set2')
plt.title('Wykres pudełkowy residuów dla {} próbek zbioru testowego'.format(test_len))
plt.ylabel('Moc [MW]')
f.savefig('./plots/boxplot.pdf', bbox_inches='tight')
plt.show()

# get coeffs without const
coefficients_list = [[1] + [*d.values][1:] for d in coefficients]
# get info about roots
roots = [get_k_roots_with_largest_radiuses(r, NR_OF_PHASES_TO_PLOT) for r in coefficients_list]

# plot the phases (from -180 to 180 degrees) of polynomial roots over time (roots are sorted by non-increasing radius)
f = plt.figure(figsize=(12, 12))
for i in range(NR_OF_PHASES_TO_PLOT):
    were_original_radiuses_in_unit_circle = [pred[i]['was_original_radius_in_unit_circle'] for pred in roots]
    phases = [pred[i]['phase'] for pred in roots]
    plt.subplot(NR_OF_PHASES_TO_PLOT, 1, i + 1)
    for j, (phase, was_original_radius_in_unit_circle) in enumerate(zip(phases, were_original_radiuses_in_unit_circle)):
        if was_original_radius_in_unit_circle:
            colour = 'bo'
        else:
            colour = 'ro'
        plt.plot(j + 1, phase, colour)
    plt.grid()
    plt.xticks(range(1, len(roots) + 1))
plt.xlabel('Numer próbki')
f.savefig('./plots/phases.pdf', bbox_inches='tight')
plt.show()
