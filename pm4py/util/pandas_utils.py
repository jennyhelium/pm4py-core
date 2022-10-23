'''
    This file is part of PM4Py (More Info: https://pm4py.fit.fraunhofer.de).

    PM4Py is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PM4Py is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with PM4Py.  If not, see <https://www.gnu.org/licenses/>.
'''
import pkgutil

import pandas as pd

from pm4py.util import constants, xes_constants
import numpy as np


def to_dict_records(df):
    """
    Pandas dataframe to dictionary (records method)

    Parameters
    ---------------
    df
        Dataframe

    Returns
    --------------
    list_dictio
        List containing a dictionary for each row
    """
    return df.to_dict('records')


def to_dict_index(df):
    """
    Pandas dataframe to dictionary (index method)

    Parameters
    ---------------
    df
        Dataframe

    Returns
    --------------
    dict
        dict like {index -> {column -> value}}
    """
    return df.to_dict('index')


def insert_index(df, column_name=constants.DEFAULT_INDEX_KEY, copy_dataframe=True):
    """
    Inserts the dataframe index in the specified column

    Parameters
    --------------
    df
        Dataframe
    column_name
        Name of the column that should host the index
    copy_dataframe
        Establishes if the original dataframe should be copied before inserting the column

    Returns
    --------------
    df
        Dataframe with index
    """
    if copy_dataframe:
        df = df.copy()
    df = df.reset_index(drop=True)
    df[column_name] = df.index
    return df


def insert_case_index(df, column_name=constants.DEFAULT_CASE_INDEX_KEY, case_id=constants.CASE_CONCEPT_NAME, copy_dataframe=True):
    """
    Inserts the case number in the dataframe

    Parameters
    ---------------
    df
        Dataframe
    column_name
        Name of the column that should host the case index
    case_id
        Case identifier
    copy_dataframe
        Establishes if the original dataframe should be copied before inserting the column

    Returns
    ---------------
    df
        Dataframe with case index
    """
    if copy_dataframe:
        df = df.copy()
    df[column_name] = df.groupby(case_id).ngroup()
    return df


def insert_ev_in_tr_index(df: pd.DataFrame, case_id: str = constants.CASE_CONCEPT_NAME,
                          column_name: str = constants.DEFAULT_INDEX_IN_TRACE_KEY) -> pd.DataFrame:
    """
    Inserts a column that specify the index of the event inside the case

    Parameters
    ---------------
    df
        Dataframe
    case_id
        Column that hosts the case identifier
    column_name
        Name of the column that should host the index

    Returns
    --------------
    df
        Dataframe with index
    """
    df = df.copy()
    df_trace_idx = df.groupby(case_id).cumcount()
    df[column_name] = df_trace_idx
    return df


def insert_feature_activity_position_in_trace(df: pd.DataFrame, case_id: str = constants.CASE_CONCEPT_NAME,
                                      activity_key: str = xes_constants.DEFAULT_NAME_KEY, prefix="@@position_"):
    """
    Inserts additional columns @@position_ACT1, @@position_ACT2 ...
    which are populated for every event having activity ACT1, ACT2 respectively,
    with the index of the event inside its case.

    Parameters
    ------------------
    df
        Pandas dataframe
    case_id
        Case idntifier
    activity_key
        Activity
    prefix
        Prefix of the "activity position in trace" feature (default: @@position_)

    Returns
    ------------------
    df
        Pandas dataframe
    """
    df = insert_ev_in_tr_index(df, case_id=case_id)
    activities = set(df[activity_key].unique())
    for act in activities:
        df[prefix + act] = df[activity_key].apply(lambda x: np.nan if x == act else -1)
        df[prefix + act] = df[prefix + act].fillna(df[constants.DEFAULT_INDEX_IN_TRACE_KEY])
        df[prefix + act] = df[prefix + act].replace(-1, np.nan)
    return df


def check_is_pandas_dataframe(log):
    """
    Checks if a log object is a dataframe

    Parameters
    -------------
    log
        Log object

    Returns
    -------------
    boolean
        Is dataframe?
    """
    if pkgutil.find_loader("pandas"):
        import pandas as pd
        return type(log) is pd.DataFrame
    return False


def check_pandas_dataframe_columns(df, activity_key=None, case_id_key=None, timestamp_key=None):
    """
    Checks if the dataframe contains all the required columns.
    If not, raise an exception

    Parameters
    --------------
    df
        Pandas dataframe
    """
    if len(df.columns) < 3:
        raise Exception("the dataframe should (at least) contain a column for the case identifier, a column for the activity and a column for the timestamp.")

    str_columns = {x for x in df.columns if "str" in str(df[x].dtype).lower() or "obj" in str(df[x].dtype).lower()}
    timest_columns = {x for x in df.columns if "date" in str(df[x].dtype).lower() or "time" in str(df[x].dtype).lower()}

    if len(str_columns) < 2:
        raise Exception("the dataframe should (at least) contain a column of type string for the case identifier and a column of type string for the activity.")

    if len(timest_columns) < 1:
        raise Exception("the dataframe should (at least) contain a column of type date")

    if case_id_key is not None:
        if case_id_key not in df.columns:
            raise Exception("the specified case ID column is not contained in the dataframe. Available columns: "+str(sorted(list(df.columns))))

        if case_id_key not in str_columns:
            raise Exception("the case ID column should be of type string.")

        if df[case_id_key].isnull().values.any():
            raise Exception("the case ID column should not contain any empty value.")

    if activity_key is not None:
        if activity_key not in df.columns:
            raise Exception("the specified activity column is not contained in the dataframe. Available columns: "+str(sorted(list(df.columns))))

        if activity_key not in str_columns:
            raise Exception("the activity column should be of type string.")

        if df[activity_key].isnull().values.any():
            raise Exception("the activity column should not contain any empty value.")

    if timestamp_key is not None:
        if timestamp_key not in df.columns:
            raise Exception("the specified timestamp column is not contained in the dataframe. Available columns: "+str(sorted(list(df.columns))))

        if timestamp_key not in timest_columns:
            raise Exception("the timestamp column should be of time datetime. Use the function pandas.to_datetime")

        if df[timestamp_key].isnull().values.any():
            raise Exception("the timestamp column should not contain any empty value.")

    """if len(set(df.columns).intersection(
            set([constants.CASE_CONCEPT_NAME, xes_constants.DEFAULT_NAME_KEY,
                 xes_constants.DEFAULT_TIMESTAMP_KEY]))) < 3:
        raise Exception(
            "please format your dataframe accordingly! df = pm4py.format_dataframe(df, case_id='<name of the case ID column>', activity_key='<name of the activity column>', timestamp_key='<name of the timestamp column>')")"""
