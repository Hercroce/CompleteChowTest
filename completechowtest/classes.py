from scipy.stats import f
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import numpy as np
import matplotlib.dates as mdates


class TestReadyDF():
    def __init__(self, date_series, y, X, regression_constant=True):
        self.date_series = date_series
        self.y = y
        self.X = X
        self.regression_constant = regression_constant

    def _transform_abbreviation(self, abbreviation):
        abbreviation_mapping = {
            'y': 'year',
            'm': 'month',
            'w': 'week',
            'd': 'day',
            'h': 'hour',
            'min': 'minute',
            's': 'second'
        }

        if abbreviation in abbreviation_mapping:
            return abbreviation_mapping[abbreviation]
        else:
            raise ValueError(f"No transformation found for '{abbreviation}'")

    def add_date_columns(self, df):
        """
        Add date columns based on a df column
        """
        df = pd.DataFrame({
            'Date': df,
            'Year': df.dt.year,
            'Month': df.dt.month,
            'Day': df.dt.day
        })
        return df

    def slice_dates(self, df, test_incidence_unit, start_end='end'):

        test_incidence_unit = test_incidence_unit.lower()

        if test_incidence_unit not in ('y', 'm', 'd', 'h', 'minute', 's'):
            raise ValueError("Please choose 'y', 'm', 'd', 'h', 'minute' or 's'.")

        dates = self.add_date_columns(df)

        if start_end == 'start':
            start_end = 'min'
        elif start_end == 'end':
            start_end = 'max'
        else:
            raise ValueError("Only 'start' and 'end' are acceptable")

        if test_incidence_unit == 'y':
            dates_indexes = dates.groupby(["Year"], as_index=False).agg({'Date': start_end})
            return dates_indexes

        elif test_incidence_unit == 'm':
            dates_indexes = dates.groupby(["Year", "Month"], as_index=False).agg({'Date': start_end})
            return dates_indexes

        elif test_incidence_unit == 'd':
            dates_indexes = dates.groupby(["Year", "Month", 'Day'], as_index=False).agg({'Date': start_end})
            return dates_indexes

        elif test_incidence_unit == 'h':
            dates_indexes = dates.groupby(["Year", "Month", 'Day', 'Hour'], as_index=False).agg({'Date': start_end})
            return dates_indexes

        elif test_incidence_unit == 'minute':
            dates_indexes = dates.groupby(["Year", "Month", 'Day', 'Hour', 'Minute'], as_index=False).agg(
                {'Date': start_end})
            return dates_indexes

        elif test_incidence_unit == 'second':
            dates_indexes = dates.groupby(["Year", "Month", 'Day', 'Hour', 'Minute', 'Second'], as_index=False).agg(
                {'Date': start_end})
            return dates_indexes

    def _linear_residuals(self, y, X):
        if self.regression_constant:
            X = sm.add_constant(X)

        model = sm.OLS(y, X)
        results = model.fit()

        summary_result = pd.DataFrame()
        summary_result['y_actual'] = y
        summary_result['y_hat'] = results.predict(X)
        summary_result['residuals'] = results.resid
        summary_result['residuals_sq'] = results.resid ** 2

        return summary_result

    def _calculate_RSS(self, y, X):
        # calls the linear_residual function
        resid_data = self._linear_residuals(y, X)
        # calculates the sum of squared resiudals
        rss = resid_data.residuals_sq.sum()

        # returns the sum of squared residuals
        return (rss)

    # defines a function to return the p-value from a Chow Test
    def chow_test(self, y, X, last_index_in_model_1, first_index_in_model_2):
        # gets the RSS for the entire period
        rss_pooled = self._calculate_RSS(y, X)

        # splits the X and y dataframes and gets the rows from the first row in the dataframe
        # to the last row in the model 1 testing period and then calculates the RSS
        X1 = X.loc[:last_index_in_model_1]
        y1 = y.loc[:last_index_in_model_1]
        rss1 = self._calculate_RSS(y1, X1)

        # splits the X and y dataframes and gets the rows from the first row in the model 2
        # testing period to the last row in the dataframe and then calculates the RSS
        X2 = X.loc[first_index_in_model_2:]
        y2 = y.loc[first_index_in_model_2:]
        rss2 = self._calculate_RSS(y2, X2)

        # gets the number of independent variables, plus 1 for the constant in the regression
        ##### Mudar essa soma no caso de não constante.
        k = X.shape[1]
        # gets the number of observations in the first period
        N1 = X1.shape[0]
        # gets the number of observations in the second period
        N2 = X2.shape[0]

        # calculates the numerator of the Chow Statistic
        numerator = (rss_pooled - (rss1 + rss2)) / k
        # calculates the denominator of the Chow Statistic
        denominator = (rss1 + rss2) / (N1 + N2 - 2 * k)

        # calculates the Chow Statistic
        Chow_Stat = numerator / denominator

        # calculates the p-value by subtracting 1 by the cumulative probability at the Chow
        # statistic from an F-distribution with k and N1 + N2 - 2k degrees of freedom
        p_value = 1 - f.cdf(Chow_Stat, dfn=k, dfd=(N1 + N2 - 2 * k))

        # saves the Chow_State and p_value in a tuple
        result = (Chow_Stat, p_value)

        # returns the p-value
        return (result)

    def comprehensive_table(self, test_incidence_unit, test_span_unit):
        draft_Chow_tests = []
        adf_tests = {}
        test_incidence_unit = self._transform_abbreviation(test_incidence_unit)

        start_dates = self.slice_dates(self.date_series, test_span_unit, 'start')
        end_dates = self.slice_dates(self.date_series, test_span_unit, 'end')

        base_df = pd.DataFrame({'Date': self.date_series})
        base_df = pd.concat([base_df, self.y, self.X], axis=1)

        for start_date in start_dates['Date']:
            for end_date in end_dates['Date']:
                # Aqui separei por ano, tenho que separar a granulidade.
                start_incidence_unit = getattr(start_date, test_incidence_unit)
                end_incidence_unit = getattr(end_date, test_incidence_unit)

                if start_incidence_unit < end_incidence_unit:

                    main_df = base_df.query("Date >= @start_date and Date <= @end_date")
                    slice_dates = start_dates.query("Date > @start_date and Date < @end_date")
                    y = main_df[["Y"]]
                    X = main_df[["X1", "X2"]]
                    pool_observations = main_df.shape[0]

                    # ADF tests
                    y_series = y.iloc[:, 0]
                    adf_tests[f'ADF Statistic {y_series.name}'] = adfuller(y_series)[0]
                    adf_tests[f'ADF p-value {y_series.name}'] = adfuller(y_series)[1]

                    n_exo_vars = X.shape[1]
                    for var in range(0, n_exo_vars, 1):
                        series = X.iloc[:, var]

                        adf_tests[f'ADF Statistic {series.name}'] = adfuller(series)[0]
                        adf_tests[f'ADF p-value {series.name}'] = adfuller(series)[1]

                    # Chow tests
                    for slice_date in slice_dates['Date']:
                        first_model_2_index = main_df[main_df['Date'] == slice_date].index[0]
                        last_model_1_index = first_model_2_index - 1

                        subsample_a_observations = y.loc[:last_model_1_index].shape[0]
                        subsample_b_observations = y.loc[first_model_2_index:].shape[0]

                        sample_chow_test = self.chow_test(y, X, last_model_1_index, first_model_2_index)
                        chow_statistic = sample_chow_test[0]
                        chow_p_value = sample_chow_test[1]
                        draft_Chow_tests.append([start_date
                                                    , end_date
                                                    , slice_date
                                                    , pool_observations
                                                    , subsample_a_observations
                                                    , subsample_b_observations
                                                    , chow_statistic
                                                    , chow_p_value
                                                    , *adf_tests.values()])

        draft_chow_columns = ['Start'
            , 'End'
            , 'Slice'
            , 'Pool Observations'
            , 'Sumbsample A observations'
            , 'Sumbsample B observations'
            , 'Chow Test Statistic'
            , 'Chow Test p-value'
            , *adf_tests]

        draft_Chow_tests = pd.DataFrame(draft_Chow_tests, columns=draft_chow_columns)

        return draft_Chow_tests
