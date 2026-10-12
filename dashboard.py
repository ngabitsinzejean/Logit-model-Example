KeyError: This app has encountered an error. The original error message is redacted to prevent data leaks. Full error details have been recorded in the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app).
Traceback:
File "/home/adminuser/venv/lib/python3.14/site-packages/pandas/core/indexes/base.py", line 3641, in get_loc
    return self._engine.get_loc(casted_key)
           ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^
File "pandas/_libs/index.pyx", line 168, in pandas._libs.index.IndexEngine.get_loc
File "pandas/_libs/index.pyx", line 197, in pandas._libs.index.IndexEngine.get_loc
File "pandas/_libs/hashtable_class_helper.pxi", line 7668, in pandas._libs.hashtable.PyObjectHashTable.get_item
File "pandas/_libs/hashtable_class_helper.pxi", line 7676, in pandas._libs.hashtable.PyObjectHashTable.get_item
KeyError
The above exception was the direct cause of the following exception:
File "/mount/src/car-ownership-dashboard/dashboard.py", line 59, in <module>
    df_matrix['1 - Pi'] = 1 - df_matrix['Pi']
                              ~~~~~~~~~^^^^^^
File "/home/adminuser/venv/lib/python3.14/site-packages/pandas/core/frame.py", line 4378, in __getitem__
    indexer = self.columns.get_loc(key)
File "/home/adminuser/venv/lib/python3.14/site-packages/pandas/core/indexes/base.py", line 3648, in get_loc
    raise KeyError(key) from err
