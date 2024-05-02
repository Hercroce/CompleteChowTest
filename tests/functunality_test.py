from base_folder.completechowtest.classes import TestReadyDF
import numpy as np
import pandas as pd



def test_comprehensive_table():
    # Generate random numbers
    draft_dev_df = np.random.uniform(0, 100, size=(3000, 3))
    column_names = ["Y", 'X1', 'X2']
    draft_dev_df = pd.DataFrame(draft_dev_df, columns=column_names)
    draft_dev_df['Date'] = pd.date_range("2018-01-01", periods=3000, freq="d")

    y = draft_dev_df[['Y']]
    X = draft_dev_df[['X1', 'X2']]

    test_df = TestReadyDF(draft_dev_df,y,X)

    results = test_df.comprehensive_table(test_incidence_unit='y', test_span_unit='y')

    return results


results = test_comprehensive_table()
print(results)