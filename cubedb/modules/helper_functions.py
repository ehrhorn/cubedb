from datetime import datetime
from inspect import currentframe, getframeinfo
import json
from pathlib import Path
import pickle
import re
from typing import List, Dict
import shelve
from shutil import copyfile, move
import sqlite3
from typing import NoReturn

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


def open_pickle_file(path):
    with open(str(path), "rb") as f:
        output = pickle.load(f)
    return output


def save_pickle_file(path, item):
    with open(str(path), "wb") as f:
        pickle.dump(item, f, protocol=4)


def open_json_file(path):
    with open(str(path), "r") as f:
        output = json.load(f)
    return output


def save_json_file(path, item):
    with open(str(path), "w") as f:
        json.dump(item, f)


def sqlite_to_numpy(path, keys, table, output_event_nos=False, input_event_nos=None):
    if isinstance(keys, str):
        keys = [keys]
    if output_event_nos:
        keys = ["event_no"] + keys
    if input_event_nos is not None:
        query = "select {} from '{}' where event_no in ({})".format(
            ", ".join([key for key in keys]),
            table,
            ", ".join([str(event_no) for event_no in input_event_nos]),
        )
    else:
        query = "select {} from '{}'".format(", ".join([key for key in keys]), table)
    with sqlite3.connect(str(path)) as con:
        cur = con.cursor()
        cur.execute(query)
        data = np.array(cur.fetchall())
    return data


def sqlite_to_numpy_condition(
    path, keys, table, condition_key, conditions, with_event_no=False
):
    if isinstance(keys, str):
        keys = [keys]
    if with_event_no:
        keys = ["event_no"] + keys
    query = "select {} from '{}' where {} = ('{}')".format(
        ", ".join([key for key in keys]), table, condition_key, ", ".join(conditions)
    )
    with sqlite3.connect(str(path)) as con:
        cur = con.cursor()
        cur.execute(query)
        data = np.array(cur.fetchall())
    return data


def sqlite_to_numpy_generator(
    path,
    keys,
    table,
    input_event_nos=False,
    n=100,
    output_event_no=False,
    output_row_no=False,
):
    if isinstance(keys, str):
        keys = [keys]
    if output_event_no:
        keys = ["event_no"] + keys
    if output_row_no:
        keys = ["row"] + keys
    if input_event_nos:
        event_nos = [str(event_no) for event_no in input_event_nos]
        query = "select {} from '{}' where event_no in ({})".format(
            ", ".join([key for key in keys]), table, ", ".join(event_nos)
        )
    else:
        query = "select {} from '{}'".format(", ".join([key for key in keys]), table)
    with sqlite3.connect(str(path)) as con:
        cur = con.cursor()
        cur.execute(query)
        while True:
            results = cur.fetchmany(n)
            if not results:
                break
            else:
                yield np.array(results)


def sqlite_to_numpy_generator_event_length(
    path,
    keys,
    table,
    input_event_nos,
    n=100,
    output_event_no=False,
    output_row_no=False,
):
    if isinstance(keys, str):
        keys = [keys]
    if output_event_no:
        keys = ["event_no"] + keys
    if output_row_no:
        keys = ["row"] + keys
    event_nos = [str(event_no) for event_no in input_event_nos]
    query = "select sum({}) from '{}' where event_no in ({}) group by event_no".format(
        ", ".join([key for key in keys]), table, ", ".join(event_nos)
    )
    with sqlite3.connect(str(path)) as con:
        cur = con.cursor()
        cur.execute(query)
        while True:
            results = cur.fetchmany(n)
            if not results:
                break
            else:
                yield np.array(results)


def sqlite_to_numpy_pandas(path, keys, table, with_event_no=False):
    if isinstance(keys, str):
        keys = [keys]
    if with_event_no:
        keys = ["event_no"] + keys
    query = "select {} from '{}'".format(", ".join([key for key in keys]), table)
    with sqlite3.connect(str(path)) as con:
        data = pd.read_sql(query, con).values
    return data


def sqlite_get_event_nos(path):
    query = "select distinct(event_no) from features"
    with sqlite3.connect(str(path)) as con:
        cur = con.cursor()
        cur.execute(query)
        event_nos = np.array(cur.fetchall())
    return event_nos


def choose_n_random_event_nos(event_nos, size=10000):
    if size > len(event_nos):
        size = len(event_nos)
    np.random.seed(seed=29897070)
    choices = np.random.choice(event_nos.flatten(), size=size, replace=False)
    return choices


def sqlite_to_numpy_random(path, keys, table, size, with_event_no=False):
    event_nos = sqlite_get_event_nos(path)
    event_nos = choose_n_random_event_nos(event_nos, size).flatten().tolist()
    if isinstance(keys, str):
        keys = [keys]
    if with_event_no:
        keys = ["event_no"] + keys
    query = "select {} from '{}' where event_no in ({})".format(
        ", ".join(keys), table, ", ".join(map(str, event_nos))
    )
    with sqlite3.connect(str(path)) as con:
        cur = con.cursor()
        cur.execute(query)
        data = np.array(cur.fetchall())
    return data


def fit_transformer(data, transformer):
    data = data.flatten().reshape(-1, 1)
    transformer = transformer.fit(data)
    return transformer


def transform_data(data, transformer):
    data = data.flatten().reshape(-1, 1)
    transformed_data = transformer.transform(data).flatten()
    return transformed_data


def inverse_transform_data(data, transformer):
    data = data.flatten().reshape(-1, 1)
    inverse_transformed_data = transformer.inverse_transform(data).flatten()
    return inverse_transformed_data


def create_transformers(dataset_name: str):
    paths = get_dataset_paths(dataset_name)
    if paths["transformers"].is_file():
        print_message_with_time("Transformers already created")
        return
    else:
        now = datetime.now()
        print("{}: Beginning transformation fitting".format(now))
        features = {
            "dom_x": RobustScaler,
            "dom_y": RobustScaler,
            "dom_z": RobustScaler,
            "dom_r": RobustScaler,
            "dom_zenith": RobustScaler,
            "dom_azimuth": RobustScaler,
            "pmt_zenith": RobustScaler,
            "pmt_azimuth": RobustScaler,
            "time": RobustScaler,
            "charge_log10": RobustScaler,
        }
        truths = {
            "energy_log10": RobustScaler,
            "time": RobustScaler,
            "direction_x": RobustScaler,
            "direction_y": RobustScaler,
            "direction_z": RobustScaler,
            "position_x": RobustScaler,
            "position_y": RobustScaler,
            "position_z": RobustScaler,
            "azimuth": RobustScaler,
            "zenith": RobustScaler,
        }
        transformers_dict = {}
        transformers_dict["features"] = {}
        transformers_dict["truth"] = {}
        for feature in features:
            now = datetime.now()
            print("{}: Fitting transformer on {}".format(now, feature))
            data = sqlite_to_numpy_random(
                paths["fast_db"], feature, "features", size=100000
            )
            transformer = fit_transformer(data, features[feature]())
            transformers_dict["features"][feature] = transformer
        try:
            for truth in truths:
                now = datetime.now()
                print("{}: Fitting transformer on {}".format(now, truth))
                data = sqlite_to_numpy_random(
                    paths["fast_db"], truth, "truth", size=10000
                )
                transformer = fit_transformer(data, truths[truth]())
                transformers_dict["truth"][truth] = transformer
        except Exception as e:
            print("{}: {}".format(datetime.now(), str(e)))
        save_pickle_file(paths["transformers"], transformers_dict)
        now = datetime.now()
        print("{}: Ended fitting transformers".format(now))


def create_sets(dataset_name: str):
    paths = get_dataset_paths(dataset_name)
    if paths["sets"].is_file():
        print_message_with_time("Sets already created")
        return
    else:
        now = datetime.now()
        print("{}: Beginning set creation for dataset {}".format(now, dataset_name))
        query = "select distinct(event_no) from features"
        with sqlite3.connect(str(paths["fast_db"])) as con:
            event_nos = pd.read_sql(query, con)["event_no"].values.tolist()
        sets = {}
        train, val = train_test_split(event_nos, test_size=0.2, random_state=29897070)
        sets["train"] = train
        sets["val"] = val
        sets["test"] = train + val
        intersection = set(train) & set(val)
        now = datetime.now()
        print("{}: The intersection between sets is {}".format(now, len(intersection)))
        for key in sets:
            print(
                "{}: {} contains {} events".format(datetime.now(), key, len(sets[key]))
            )
        save_pickle_file(paths["sets"], sets)


def create_run_paths(run_name):
    paths = {}
    runs_path = Path().home().joinpath("work").joinpath("runs")
    ready_path = runs_path.joinpath("ready")
    training_path = runs_path.joinpath("training")
    trained_path = runs_path.joinpath("trained")
    done_path = runs_path.joinpath("done")

    ready_path.mkdir(parents=True, exist_ok=True)
    training_path.mkdir(parents=True, exist_ok=True)
    trained_path.mkdir(parents=True, exist_ok=True)
    done_path.mkdir(parents=True, exist_ok=True)

    ready_path = ready_path.joinpath(run_name)
    training_path = training_path.joinpath(run_name)
    trained_path = trained_path.joinpath(run_name)
    done_path = done_path.joinpath(run_name)

    paths["ready"] = ready_path
    paths["training"] = training_path
    paths["trained"] = trained_path
    paths["done"] = done_path

    for path, values in paths.items():
        root_path = paths[path]
        paths[path] = {}
        paths[path]["root"] = root_path
        paths[path]["weights"] = root_path.joinpath("weights.h5")
        paths[path]["params"] = root_path.joinpath("params.json")
        paths[path]["csv"] = root_path.joinpath("log.csv")
        paths[path]["lr"] = root_path.joinpath("lr_history.pkl")
        paths[path]["summary"] = root_path.joinpath("summary.txt")
    return paths


def create_run_dir(paths):
    paths["root"].mkdir(parents=True, exist_ok=True)


def get_dataset_paths(dataset_name: str) -> Dict[str, Path]:
    paths = {}
    home = Path().home()
    data_dir = Path().home().joinpath("data")
    fast_db = data_dir.joinpath(dataset_name + ".db")
    dataset_path = home.joinpath("work").joinpath("datasets").joinpath(dataset_name)
    data_path = dataset_path.joinpath("data")
    meta_path = dataset_path.joinpath("meta")
    data_path.mkdir(parents=True, exist_ok=True)
    meta_path.mkdir(parents=True, exist_ok=True)
    slow_db = data_path.joinpath(dataset_name + ".db")
    transformer_file = meta_path.joinpath("transformers.pkl")
    sets_file = meta_path.joinpath("sets.pkl")
    distributions_file = meta_path.joinpath("distributions.pkl")
    paths["fast_db"] = fast_db
    paths["db"] = slow_db
    paths["meta"] = meta_path
    paths["transformers"] = transformer_file
    paths["sets"] = sets_file
    paths["distributions"] = distributions_file
    errors_path = dataset_path.joinpath("errors")
    errors_path.mkdir(parents=True, exist_ok=True)
    errors_db = errors_path.joinpath("errors.db")
    paths["errors"] = errors_db
    predictions_path = dataset_path.joinpath("predictions")
    predictions_path.mkdir(parents=True, exist_ok=True)
    predictions_db = predictions_path.joinpath("predictions.db")
    paths["predictions"] = predictions_db
    return paths


def check_numpy_dataset_exists(paths):
    if paths["np_data"].is_file():
        return True
    else:
        return False


def create_numpy_dataset_path(dataset_name, dataset_type, paths):
    paths["np_data"].mkdir(parents=True, exist_ok=True)


def print_message_with_time(message: str) -> None:
    now = datetime.now()
    message = ("{}: " + message).format(now)
    print(message)


def sql_executemany(path, query, values):
    with sqlite3.connect(str(path)) as con:
        cur = con.cursor()
        cur.executemany(query, ((val,) for val in values))


def sql_execute(path, query):
    with sqlite3.connect(str(path)) as con:
        cur = con.cursor()
        cur.execute(query)


def sql_update_values(path, key, table, data):
    if table == "features":
        id_column = "row"
    else:
        id_column = "event_no"
    records_to_update = [(data[i, 1], int(data[i, 0])) for i in range(data.shape[0])]
    with sqlite3.connect(str(path)) as con:
        query = "update {} set {} = ? where {} = ?".format(
            table,
            key,
            id_column,
        )
        cur = con.cursor()
        cur.executemany(query, records_to_update)
        con.commit()


def transform_db(dataset_name: str, n: int = 100000):
    paths = get_dataset_paths(dataset_name)
    transformers = open_pickle_file(paths["transformers"])
    for table in transformers:
        if table == "features":
            id_column = "row"
        else:
            id_column = "event_no"
        for key, transformer in transformers[table].items():
            now = datetime.now()
            print("{}: Transforming {}".format(now, key))
            query_1 = "select {} from '{}'".format(
                ", ".join([id_column] + [key]), table
            )
            with sqlite3.connect(str(paths["fast_db"])) as con:
                cur1 = con.cursor()
                cur1.execute(query_1)
                while True:
                    data = cur1.fetchmany(n)
                    if not data:
                        break
                    data = np.array(data)
                    ids = data[:, 0].astype(int)
                    values = data[:, 1]
                    values = transformer.transform(values.reshape(-1, 1)).flatten()
                    records_to_update = [
                        (float(values[i]), int(ids[i])) for i in range(data.shape[0])
                    ]
                    query_2 = "update {} set {} = ? where {} = ?".format(
                        table,
                        key,
                        id_column,
                    )
                    cur2 = con.cursor()
                    cur2.executemany(query_2, records_to_update)
                    con.commit()


def index_db(dataset_name: str):
    paths = get_dataset_paths(dataset_name)
    indexing_queries = {}
    indexing_queries[
        "features_event_no"
    ] = """
        CREATE INDEX index_features_event_no ON features(event_no);
    """
    # indexing_queries[
    #     "truth"
    # ] = """
    #     CREATE UNIQUE INDEX index_truth ON truth(event_no);
    # """
    # indexing_queries[
    #     "meta"
    # ] = """
    #     CREATE UNIQUE INDEX index_meta ON meta(event_no);
    # """
    print_message_with_time("indexing DB")
    for query in indexing_queries.values():
        try:
            sql_execute(paths["fast_db"], query)
        except Exception:
            print_message_with_time("Index already exists or table doesn't exist")


def sql_get_table_names(path):
    query = "select name from sqlite_master where type='table'"
    with sqlite3.connect(str(path)) as con:
        cur = con.cursor()
        cur.execute(query)
        values = cur.fetchall()
    values = [value[0] for value in values]
    return values


def sql_get_cleaned_column_names(path, table):
    drop_columns = ["row", "pulse_no", "event_no"]
    query = "PRAGMA table_info({})".format(table)
    with sqlite3.connect(str(path)) as con:
        cur = con.cursor()
        cur.execute(query)
        values = cur.fetchall()
    values = [value[1] for value in values if value[1] not in drop_columns]
    return values


def sqlite_get_primary_key(path, table):
    table_info_query = "pragma table_info({})".format(table)
    with sqlite3.connect(str(path)) as con:
        cur = con.cursor()
        cur.execute(table_info_query)
        table_info = cur.fetchall()
    for row in table_info:
        if row[5] == 1:
            primary_key = row[1]
    return primary_key


def sqlite_get_min_max_n(path, column, table):
    primary_key = sqlite_get_primary_key(path, table)
    min_query = "select min({}) from '{}'".format(column, table)
    max_query = "select max({}) from '{}'".format(column, table)
    n_query = "select count({}) from '{}'".format(primary_key, table)
    with sqlite3.connect(str(path)) as con:
        cur = con.cursor()
        cur.execute(min_query)
        min_value = np.array(cur.fetchone())[0]
        cur.execute(max_query)
        max_value = np.array(cur.fetchone())[0]
        cur.execute(n_query)
        n = np.array(cur.fetchone())[0]
    return min_value, max_value, n


def sqlite_get_min_max_event_length(path, column, table):
    min_query = """
        SELECT
            sum({}) AS event_length
        FROM
            {}
        GROUP BY
            event_no
        ORDER BY
            event_length ASC
        LIMIT 1
    """.format(
        column, table
    )
    max_query = """
        SELECT
            sum({}) AS event_length
        FROM
            {}
        GROUP BY
            event_no
        ORDER BY
            event_length DESC
        LIMIT 1
    """.format(
        column, table
    )
    n_query = """
        SELECT
            count(DISTINCT (event_no))
        FROM
            features
    """
    with sqlite3.connect(str(path)) as con:
        cur = con.cursor()
        cur.execute(min_query)
        min_value = np.array(cur.fetchone())[0]
        cur.execute(max_query)
        max_value = np.array(cur.fetchone())[0]
        cur.execute(n_query)
        n = np.array(cur.fetchone())[0]
    return min_value, max_value, n


def sqlite_get_unique(path, column, table):
    query = "select distinct({}) from '{}'".format(column, table)
    try:
        with sqlite3.connect(str(path)) as con:
            cur = con.cursor()
            cur.execute(query)
            unique = sorted(np.array(cur.fetchall()).flatten().tolist())
    except Exception as e:
        print("{}: Error encountered: {}".format(datetime.now(), str(e)))
        return
    return unique


def create_dataset_distribution_histograms(dataset_name: str, transform: bool = False):
    bar_chart_columns = [
        "string",
        "dom",
        "pmt",
        "pmt_x",
        "pmt_y",
        "pmt_z",
        "pmt_area",
        "pmt_type",
        "lc",
        "atwd",
        "fadc",
        "pulse_width",
        "pid",
        "interaction_type",
        "stopped_muon",
    ]
    paths = get_dataset_paths(dataset_name)
    if transform:
        transform_type = "transformed"
        transformers = open_pickle_file(paths["transformers"])
    else:
        transform_type = "raw"
        transformers = None
    if paths["distributions"].is_file():
        hist_dict = open_pickle_file(paths["distributions"])
    else:
        hist_dict = {}
    hist_dict[transform_type] = {}
    sets = open_pickle_file(paths["sets"])
    drop_tables = ["meta"]
    tables = sql_get_table_names(paths["fast_db"])
    tables = [table for table in tables if table not in drop_tables]
    for table in tables:
        hist_dict[transform_type][table] = {}
        columns = sql_get_cleaned_column_names(paths["fast_db"], table)
        for column in columns:
            if transformers is not None and column not in transformers[table]:
                continue
            now = datetime.now()
            print("{}: Calculating histograms for {}".format(now, column))
            if column in bar_chart_columns:
                temp, ok_bool = calculate_bar_chart(
                    paths["fast_db"], column, table, sets
                )
                if ok_bool:
                    hist_dict[transform_type][table][column] = {}
                    hist_dict[transform_type][table][column] = temp
            else:
                temp, ok_bool = calculate_1d_histogram(
                    paths["fast_db"], column, table, sets
                )
                if ok_bool:
                    hist_dict[transform_type][table][column] = {}
                    hist_dict[transform_type][table][column] = temp
    save_pickle_file(paths["distributions"], hist_dict)


def calculate_bar_chart(path, column, table, sets):
    hist_dict = {}
    unique = sqlite_get_unique(path, column, table)
    if unique is not None:
        for iset in sets.keys():
            hist_dict[iset] = {}
            hist_dict[iset]["bin_edges"] = unique
            hist_dict[iset]["hist"] = [0 for i in unique]
            hist_dict[iset]["events"] = len(sets[iset])
            hist_dict[iset]["type"] = "bar"
            generator = sqlite_to_numpy_generator(
                path, column, table, sets[iset], n=10000
            )
            for data in generator:
                try:
                    data = data[np.isfinite(data)]
                    for i, value in enumerate(unique):
                        hist_dict[iset]["hist"][i] += len(data[data == value])
                except Exception:
                    continue
            if sum(hist_dict[iset]["hist"]) == 0:
                ok_bool = False
            else:
                ok_bool = True
    else:
        ok_bool = False
    return hist_dict, ok_bool


def calculate_1d_histogram(path, column, table, sets, num_bins=100, rice_bins=True):
    min_value, max_value, n = sqlite_get_min_max_n(path, column, table)
    if rice_bins:
        num_bins = int(np.ceil(2 * n ** (1 / 3)))
    hist_dict = {}
    for iset in sets.keys():
        hist_dict[iset] = {}
        try:
            hist_dict[iset]["bin_edges"] = np.linspace(min_value, max_value, num_bins)
        except Exception:
            ok_bool = False
            return hist_dict, ok_bool
        hist_dict[iset]["hist"] = np.zeros(num_bins - 1, dtype="int32")
        hist_dict[iset]["events"] = len(sets[iset])
        hist_dict[iset]["type"] = "histogram"
        if column == "SplitInIcePulses" or column == "SRTInIcePulses":
            generator = sqlite_to_numpy_generator_event_length(
                path, column, table, sets[iset], n=10000
            )
            min_value, max_value, n = sqlite_get_min_max_event_length(
                path, column, table
            )
            if rice_bins:
                num_bins = int(np.ceil(2 * n ** (1 / 3)))
            hist_dict[iset]["bin_edges"] = np.linspace(min_value, max_value, num_bins)
            hist_dict[iset]["hist"] = np.zeros(num_bins - 1, dtype="int32")
        else:
            generator = sqlite_to_numpy_generator(
                path, column, table, sets[iset], n=10000
            )
        for data in generator:
            try:
                data = data[np.isfinite(data)]
                hist_values, _ = np.histogram(data, hist_dict[iset]["bin_edges"])
                hist_dict[iset]["hist"] += hist_values
            except Exception:
                continue
        if sum(hist_dict[iset]["hist"]) == 0:
            ok_bool = False
        else:
            ok_bool = True
    return hist_dict, ok_bool


def check_if_table_empty(path, table_name):
    columns = sql_get_cleaned_column_names(path, table_name)
    data = sqlite_to_numpy(path, columns, table_name)
    if np.isnan(data).all():
        return True
    else:
        return False


def copy_reconstruction_to_prediction(dataset_name):
    print_message_with_time("Copying reconstruction to predictions")
    paths = get_dataset_paths(dataset_name)
    if paths["predictions"].is_file():
        return
    with sqlite3.connect(str(paths["db"])) as con:
        query = "select * from reconstruction"
        data = pd.read_sql(query, con)
    data.set_index("event_no", inplace=True)
    with sqlite3.connect(str(paths["predictions"])) as con:
        data.to_sql("reconstruction", con, if_exists="replace")


def bootstrap_resolution(
    data, quantiles=[0.16, 0.84], R=1000, return_bootstrap_data=False
):
    widths = []
    n = len(data)
    size = (n, R)
    sampled_data = np.random.choice(data, size=size, replace=True)
    widths = (
        np.quantile(sampled_data, quantiles[1], axis=0)
        - np.quantile(sampled_data, quantiles[0], axis=0)
    ) / 2
    width = np.mean(widths)
    std = np.std(widths)
    if not return_bootstrap_data:
        return width, std
    else:
        return width, std, (sampled_data, widths)


def estimate_resolution(data, percentiles=[16, 84], bootstrap=True, n_bootstraps=1000):
    means, plussigmas, minussigmas = estimate_percentile(
        data, percentiles, bootstrap, n_bootstraps
    )
    errors = (plussigmas - minussigmas) / 2
    factor = 1 / 1.349
    sigma = np.abs(means[1] - means[0]) * factor
    e_sigma = factor * np.sqrt(errors[0] ** 2 + errors[1] ** 2)
    return sigma, e_sigma


def convert_data_for_step(bin_edges, hist):
    widths = bin_edges[1:] - bin_edges[:-1]
    bin_edges = np.append(bin_edges, (bin_edges[-1] + widths[-1]))
    hist = np.insert(hist, 0, 0)
    hist = np.append(hist, 0)
    return bin_edges, hist


def calculate_centers_and_widths(bin_edges):
    widths = bin_edges[1:] - bin_edges[:-1]
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return centers, widths


def estimate_percentile(data, percentiles, bootstrap=True, n_bootstraps=1000):
    data = np.array(data)
    n = data.shape[0]
    data.sort()
    i_means, means = [], []
    i_plussigmas, plussigmas = [], []
    i_minussigmas, minussigmas = [], []

    for percentile in percentiles:
        sigma = np.sqrt((percentile / 100) * n * (1 - (percentile / 100)))
        mean = n * percentile / 100
        i_means.append(int(mean))
        i_plussigmas.append(int(mean + sigma + 1))
        i_minussigmas.append(int(mean - sigma))

    if bootstrap:
        bootstrap_indices = np.random.choice(np.arange(0, n), size=(n, n_bootstraps))
        bootstrap_indices.sort(axis=0)
        bootstrap_samples = data[bootstrap_indices]
        for i in range(len(i_means)):
            try:
                mean = bootstrap_samples[i_means[i], :]
                plussigma = bootstrap_samples[i_plussigmas[i], :]
                minussigma = bootstrap_samples[i_minussigmas[i], :]
                means.append(np.mean(mean))
                plussigmas.append(np.mean(plussigma))
                minussigmas.append(np.mean(minussigma))
            except IndexError:
                means.append(np.nan)
                plussigmas.append(np.nan)
                minussigmas.append(np.nan)
    else:
        for i in range(len(i_means)):
            try:
                mean = data[i_means[i]]
                plussigma = data[i_plussigmas[i]]
                minussigma = data[i_minussigmas[i]]
                means.append(mean)
                plussigmas.append(plussigma)
                minussigmas.append(minussigma)
            except IndexError:
                means.append(np.nan)
                plussigmas.append(np.nan)
                minussigmas.append(np.nan)
    return np.array(means), np.array(plussigmas), np.array(minussigmas)


def get_project_root():
    filename = getframeinfo(currentframe()).filename
    parent = Path(filename).resolve().parent.parent.parent
    return parent


def move_file(source, destination):
    move(str(source), str(destination))


def copy_file(source, destination):
    print_message_with_time("Copying file...")
    copyfile(source, destination)
    print_message_with_time("File copy complete")


def norm_histogram(data):
    counts = data["hist"]
    bins = data["bin_edges"]
    density = counts / (sum(counts) * np.diff(bins))
    return density


def norm_bar_chart(data):
    counts = data["hist"]
    density = np.array(counts) / sum(counts)
    return density


def get_i3_file_names(dataset_name):
    paths = get_dataset_paths(dataset_name)
    files = sqlite_get_unique(paths["fast_db"], "file", "meta")
    return files


def get_i3_file_number(in_file):
    file_number = re.search(r"(?<=\.|_)[0]{2}(.*)[0-9]{2}", in_file).group(0)
    return file_number


def get_i3_and_internal_event_nos(dataset_name, file_number):
    paths = get_dataset_paths(dataset_name)
    data = sqlite_to_numpy_condition(
        paths["fast_db"], ["event_no", "idx"], "meta", "file", [file_number]
    )
    return data


def match_i3_file_name(in_file, file_number):
    regex_pattern = r"(?<=\.|_)[0]{2}(.*)[0-9]{2}"
    file_name = re.sub(regex_pattern, str(file_number), in_file)
    return file_name


def save_prediction_artifacts(run_name, dataset_name, params, events, predictions):
    paths = get_dataset_paths(dataset_name)
    transformers = open_pickle_file(paths["transformers"])
    predictions_dict = {}
    predictions_dict["event_no"] = np.array(events)
    for i, target in enumerate(params["targets"]):
        if target in transformers["truth"]:
            transformer = transformers["truth"][target]
            predictions_dict[target] = transformer.inverse_transform(
                predictions[:, i].reshape(-1, 1)
            ).flatten()
        else:
            predictions_dict[target] = predictions[:, i]
    if params["type"] == "direction":
        vector = np.stack(
            (
                predictions_dict["direction_x"],
                predictions_dict["direction_y"],
                predictions_dict["direction_z"],
            ),
            axis=1,
        )
        spherical_angles = convert_cartesian_to_spherical(vector)
        predictions_dict["azimuth"] = spherical_angles[0].flatten()
        predictions_dict["zenith"] = spherical_angles[1].flatten()
    predictions_df = pd.DataFrame(data=predictions_dict)
    predictions_df.set_index("event_no", inplace=True)
    with sqlite3.connect(str(paths["predictions"])) as con:
        predictions_df.to_sql(name=str(run_name), con=con, if_exists="replace")


def calculate_errors(
    run_name: str, dataset_name: str, algorithm_types: List[str]
) -> None:
    paths = get_dataset_paths(dataset_name)
    transformers = open_pickle_file(paths["transformers"])
    query = "select * from '{}'".format(str(run_name))
    with sqlite3.connect(str(paths["predictions"])) as con:
        predictions = pd.read_sql(query, con)
    targets = predictions.columns.values.flatten().tolist()
    targets.remove("event_no")
    query = "select {} from truth".format(", ".join(["event_no"] + targets))
    with sqlite3.connect(str(paths["db"])) as con:
        truths = pd.read_sql(query, con)
    for column in truths.columns.values.tolist():
        if column != "event_no":
            if column in transformers["truth"]:
                truths[column] = inverse_transform_data(
                    truths[column].values, transformers["truth"][column]
                )
    merged = predictions.merge(
        truths, on="event_no", suffixes=("_prediction", "_truth")
    )
    errors = pd.DataFrame()
    errors["event_no"] = merged["event_no"]
    merged.set_index("event_no", inplace=True)
    errors.set_index("event_no", inplace=True)
    for algorithm_type in algorithm_types:
        if algorithm_type == "energy":
            target = "energy_log10"
            x = merged[target + "_prediction"].values.flatten()
            y = merged[target + "_truth"].values.flatten()
            errors[target] = x - y
        elif algorithm_type == "time":
            target = "time"
            x = merged[target + "_prediction"].values.flatten()
            y = merged[target + "_truth"].values.flatten()
            errors[target] = x - y
        elif algorithm_type == "direction":
            for target in [
                "azimuth",
                "zenith",
                "direction_x",
                "direction_y",
                "direction_z",
            ]:
                x = merged[target + "_prediction"].values.flatten()
                y = merged[target + "_truth"].values.flatten()
                # errors[target] = smallest_angle_difference(x, y)
                errors[target] = x - y
            errors["angle"] = angle_between(
                convert_spherical_to_cartesian(
                    merged["zenith_prediction"].values.flatten(),
                    merged["azimuth_prediction"].values.flatten(),
                ),
                convert_spherical_to_cartesian(
                    merged["zenith_truth"].values.flatten(),
                    merged["azimuth_truth"].values.flatten(),
                ),
            )
        elif algorithm_type == "vertex":
            for target in [
                "vertex_x",
                "vertex_y",
                "vertex_z",
            ]:
                x = merged[target + "_prediction"].values.flatten()
                y = merged[target + "_truth"].values.flatten()
                errors[target] = x - y
            errors["vertex"] = distance_between(
                merged[
                    [
                        "position_x_prediction",
                        "position_y_prediction",
                        "position_z_prediction",
                    ]
                ].values,
                merged[["vertex_x_truth", "vertex_y_truth", "vertex_z_truth"]].values,
            )

    with sqlite3.connect(str(paths["errors"])) as con:
        errors.to_sql(str(run_name), con, if_exists="replace")


def smallest_angle_difference(x, y):
    out = np.zeros(x.shape[0])
    a = np.mod(x - y, 2 * np.pi)
    b = np.mod(y - x, 2 * np.pi)
    for i in range(x.shape[0]):
        out[i] = -a[i] if a[i] < b[i] else b[i]
    return out


def angle_between(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Returns the angle in radians between vectors 'v1' and 'v2'.

    Accounts for opposite vectors using numpy.clip.

    Args:
        v1 (numpy.ndarray): vector 1
        v2 (numpy.ndarray): vector 2

    Returns:
        numpy.ndarray: angles between vectors 1 and 2
    """
    p1 = np.einsum("ij,ij->i", v1, v2)
    p2 = np.einsum("ij,ij->i", v1, v1)
    p3 = np.einsum("ij,ij->i", v2, v2)
    p4 = p1 / np.sqrt(p2 * p3)
    angles = np.arccos(np.clip(p4, -1.0, 1.0)).reshape(-1, 1)
    return angles


def distance_between(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Return the Euclidean distance between points 'p1' and 'p2'.

    Args:
        p1 (numpy.ndarray): point 1
        p2 (numpy.ndarray): point 2

    Returns:
        numpy.ndarray: distances between points 1 and 2
    """
    distances = np.linalg.norm((p1 - p2), axis=1).reshape(-1, 1)
    return distances


def convert_cartesian_to_spherical(vectors):
    """Convert Cartesian coordinates to signed spherical coordinates.

    Converts Cartesian vectors to unit length before conversion.

    Args:
        vectors (numpy.ndarray): x, y, z coordinates in shape (n, 3)

    Returns:
        tuple: tuple containing:
            azimuth (numpy.ndarray): signed azimuthal angles
            zenith (numpy.ndarray): zenith/polar angles
    """
    lengths = np.linalg.norm(vectors, axis=1).reshape(-1, 1)
    unit_vectors = vectors / lengths
    x = unit_vectors[:, 0]
    y = unit_vectors[:, 1]
    z = unit_vectors[:, 2]
    azimuth = np.arctan2(y, x).reshape(-1, 1) + np.pi
    zenith = np.arccos(z).reshape(-1, 1)
    return azimuth, zenith


def convert_spherical_to_cartesian(zenith, azimuth):
    """Convert spherical coordinates to Cartesian coordinates.

    Assumes unit length.

    Zenith: theta
    Azimuth: phi

    Args:
        zenith (numpy.ndarray): zenith/polar angle
        azimuth (numpy.ndarray): azimuthal angle

    Returns:
        numpy.ndarray: x, y, z (event, coordinates) vector
    """
    theta = zenith
    phi = azimuth
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    vectors = np.array((x, y, z)).T
    return vectors


def truth_to_prediction_db(dataset_name: str) -> None:
    paths = get_dataset_paths(dataset_name)
    try:
        with sqlite3.connect(str(paths["fast_db"])) as con:
            query = "select * from truth"
            truth = pd.read_sql(query, con)
    except Exception as e:
        print("{}: Error encountered: {}".format(datetime.now(), str(e)))
        return
    truth.set_index("event_no", inplace=True)
    with sqlite3.connect(str(paths["predictions"])) as con:
        truth.to_sql("truth", con, if_exists="replace")


def histogram_2d(
    x_path,
    y_path,
    x_column: str,
    y_column: str,
    x_name: str,
    y_name: str,
    hist_range: tuple,
    n_bootstraps: int,
    x_num_bins: int = 100,
    y_num_bins: int = 100,
    transpose=False,
) -> dict:
    hist_dict = {}
    hist_dict["type"] = "histogram"
    x_data = sqlite_to_numpy(x_path, x_column, x_name, output_event_nos=True)
    x_data = x_data[x_data[:, 0].argsort()]
    hist_dict["events"] = x_data.shape[0]
    y_data = sqlite_to_numpy(
        y_path,
        y_column,
        y_name,
        output_event_nos=True,
        input_event_nos=x_data[:, 0],
    )
    y_data = y_data[y_data[:, 0].argsort()]
    x_min = np.percentile(x_data[:, 1], 1)
    x_max = np.ceil(np.percentile(x_data[:, 1], 99))
    y_min = np.floor(np.amin(y_data[:, 1]))
    y_max = np.ceil(np.amax(y_data[:, 1]))
    if not transpose:
        clipped_x_data = np.clip(x_data[:, 1], y_min, y_max)
        hist_dict["bin_edges"] = np.array(
            (
                np.linspace(y_min, y_max, x_num_bins),
                np.linspace(y_min, y_max, y_num_bins),
            )
        )
    else:
        clipped_x_data = np.clip(x_data[:, 1], x_min, x_max)
        hist_dict["bin_edges"] = np.array(
            (
                np.linspace(x_min, x_max, x_num_bins),
                np.linspace(y_min, y_max, y_num_bins),
            )
        )
    clipped_y_data = np.clip(y_data[:, 1], y_min, y_max)
    if transpose:
        hist_dict["hist"], x_bins, y_bins = np.histogram2d(
            x=clipped_x_data,
            y=clipped_y_data,
            bins=[hist_dict["bin_edges"][0], hist_dict["bin_edges"][1]],
            density=True,
        )
    else:
        hist_dict["hist"], x_bins, y_bins = np.histogram2d(
            x=clipped_y_data,
            y=clipped_x_data,
            bins=[hist_dict["bin_edges"][1], hist_dict["bin_edges"][0]],
            density=True,
        )
    hist_dict["bin_edges"] = (x_bins, y_bins)
    resolution_num_bins = int(abs(y_max - y_min)) * 6 + 1
    resolution_bins = np.linspace(y_min, y_max, resolution_num_bins)
    hist_dict["lower_percentiles"] = []
    hist_dict["mid_percentiles"] = []
    hist_dict["upper_percentiles"] = []
    hist_dict["percentile_bin_mids"] = []
    hist_dict["percentile_histograms"] = {}
    for i in range(resolution_num_bins - 1):
        data = clipped_x_data[
            np.where(
                np.logical_and(
                    clipped_y_data >= resolution_bins[i],
                    clipped_y_data < resolution_bins[i + 1],
                )
            )
        ]
        percentiles = estimate_percentile(data, [16, 50, 84], n_bootstraps=n_bootstraps)
        hist_dict["lower_percentiles"].append(percentiles[0][0])
        hist_dict["mid_percentiles"].append(percentiles[0][1])
        hist_dict["upper_percentiles"].append(percentiles[0][2])
        hist_dict["percentile_bin_mids"].append(
            (resolution_bins[i] + resolution_bins[i + 1]) / 2
        )
        hist_dict["percentile_histograms"][i] = {}
        hist, bin_edges = np.histogram(data, bins=100, density=True)
        hist_dict["percentile_histograms"][i]["hist"] = hist
        hist_dict["percentile_histograms"][i]["bin_edges"] = bin_edges
        hist_dict["percentile_histograms"][i]["events"] = data.shape[0]
        hist_dict["percentile_histograms"][i]["resolution_bin"] = (
            resolution_bins[i],
            resolution_bins[i + 1],
        )

    hist_dict["lower_percentiles"] = np.array(hist_dict["lower_percentiles"])
    hist_dict["mid_percentiles"] = np.array(hist_dict["mid_percentiles"])
    hist_dict["upper_percentiles"] = np.array(hist_dict["upper_percentiles"])
    hist_dict["percentile_bin_mids"] = np.array(hist_dict["percentile_bin_mids"])
    return hist_dict


def histogram_resolution(
    x_path,
    y_path,
    x_column: str,
    y_column: str,
    x_name: str,
    y_name: str,
    hist_range: tuple,
    n_bootstraps: int,
) -> dict:
    hist_dict = {}
    hist_dict["type"] = "histogram"
    x_data = sqlite_to_numpy(x_path, x_column, x_name, output_event_nos=True)
    x_data = x_data[x_data[:, 0].argsort()]
    hist_dict["events"] = x_data.shape[0]
    y_data = sqlite_to_numpy(
        y_path,
        y_column,
        y_name,
        output_event_nos=True,
        input_event_nos=x_data[:, 0],
    )
    y_data = y_data[y_data[:, 0].argsort()]
    x_min = np.amin(x_data[:, 1])
    x_max = np.amax(x_data[:, 1])
    y_min = np.floor(np.amin(y_data[:, 1]))
    y_max = np.ceil(np.amax(y_data[:, 1]))
    clipped_x_data = np.clip(x_data[:, 1], x_min, x_max)
    clipped_y_data = np.clip(y_data[:, 1], y_min, y_max)
    resolution_num_bins = int(abs(y_max - y_min)) * 6 + 1
    resolution_bins = np.linspace(y_min, y_max, resolution_num_bins)
    hist_dict["bin_edges"] = resolution_bins
    hist_dict["hist"], _ = np.histogram(clipped_y_data, bins=resolution_num_bins)
    hist_dict["resolution"] = []
    hist_dict["resolution_sigma"] = []
    hist_dict["lower_percentiles"] = []
    hist_dict["mid_percentiles"] = []
    hist_dict["upper_percentiles"] = []
    hist_dict["percentile_bin_mids"] = []
    hist_dict["percentile_histograms"] = {}
    for i in range(resolution_num_bins - 1):
        data = clipped_x_data[
            np.where(
                np.logical_and(
                    clipped_y_data >= resolution_bins[i],
                    clipped_y_data < resolution_bins[i + 1],
                )
            )
        ]
        resolution = estimate_resolution(data, n_bootstraps=n_bootstraps)
        hist_dict["resolution"].append(resolution[0])
        hist_dict["resolution_sigma"].append(resolution[1])
        percentiles = estimate_percentile(data, [16, 50, 84], n_bootstraps=n_bootstraps)
        hist_dict["lower_percentiles"].append(percentiles[0][0])
        hist_dict["mid_percentiles"].append(percentiles[0][1])
        hist_dict["upper_percentiles"].append(percentiles[0][2])
        hist_dict["percentile_bin_mids"].append(
            (resolution_bins[i] + resolution_bins[i + 1]) / 2
        )
        hist_dict["percentile_histograms"][i] = {}
        hist, bin_edges = np.histogram(data, bins=100, density=True)
        hist_dict["percentile_histograms"][i]["hist"] = hist
        hist_dict["percentile_histograms"][i]["bin_edges"] = bin_edges
        hist_dict["percentile_histograms"][i]["events"] = data.shape[0]
        hist_dict["percentile_histograms"][i]["resolution_bin"] = (
            resolution_bins[i],
            resolution_bins[i + 1],
        )

    hist_dict["resolution"] = np.array(hist_dict["resolution"])
    hist_dict["resolution_sigma"] = np.array(hist_dict["resolution_sigma"])
    hist_dict["lower_percentiles"] = np.array(hist_dict["lower_percentiles"])
    hist_dict["mid_percentiles"] = np.array(hist_dict["mid_percentiles"])
    hist_dict["upper_percentiles"] = np.array(hist_dict["upper_percentiles"])
    hist_dict["percentile_bin_mids"] = np.array(hist_dict["percentile_bin_mids"])
    return hist_dict


def histogram_prediction(
    x_path,
    y_path,
    x_column: str,
    y_column: str,
    x_name: str,
    y_name: str,
    hist_range: tuple,
    n_bootstraps: int,
    x_num_bins: int = 100,
    y_num_bins: int = 100,
) -> dict:
    hist_dict = {}
    x_data = sqlite_to_numpy(x_path, x_column, x_name, output_event_nos=True)
    x_data = x_data[x_data[:, 0].argsort()]
    y_data = sqlite_to_numpy(
        y_path,
        y_column,
        y_name,
        output_event_nos=True,
        input_event_nos=x_data[:, 0],
    )
    y_data = y_data[y_data[:, 0].argsort()]
    clipped_x_data = np.clip(x_data[:, 1], hist_range[0][0], hist_range[0][1])
    clipped_y_data = np.clip(y_data[:, 1], hist_range[1][0], hist_range[1][1])
    hist_dict["x"] = {}
    hist_dict["y"] = {}
    hist_dict["x"]["type"] = "histogram"
    hist_dict["y"]["type"] = "histogram"
    hist_dict["x"]["events"] = x_data.shape[0]
    hist_dict["y"]["events"] = y_data.shape[0]
    hist_dict["x"]["bin_edges"] = np.linspace(
        hist_range[0][0], hist_range[0][1], x_num_bins
    )
    hist_dict["y"]["bin_edges"] = np.linspace(
        hist_range[1][0], hist_range[1][1], y_num_bins
    )
    hist_dict["x"]["hist"], _ = np.histogram(
        clipped_x_data, bins=hist_dict["x"]["bin_edges"], density=True
    )
    hist_dict["y"]["hist"], _ = np.histogram(
        clipped_y_data, bins=hist_dict["y"]["bin_edges"], density=True
    )
    return hist_dict


def prediction_2d_histogram(
    dataset_name, x_column, y_column, run, x_num_bins=100, y_num_bins=100
):
    paths = get_dataset_paths(dataset_name)
    x_min_value, x_max_value = sqlite_get_min_max(
        paths["prediction"], column, "reconstruction"
    )
    y_min_value, y_max_value = sqlite_get_min_max(paths["predictions"], column, "truth")
    hist_dict = {}
    hist_dict["bin_edges"] = (
        np.linspace(x_min_value, x_max_value, x_num_bins),
        np.linspace(y_min_value, y_max_value, y_num_bins),
    )
    hist_dict["hist"] = np.zeros(num_bins - 1, dtype="int32")
    hist_dict["events"] = 0
    hist_dict["type"] = "histogram"
    generator = sqlite_to_numpy_generator_join(
        path, x_column, y_column, "reconstruction", "truth", n=10000
    )
    for data in generator:
        data = data[np.isfinite(data)]
        hist_dict["events"] += len(data)
        hist_values, _ = np.histogramdd(data, hist_dict["bin_edges"])
        hist_dict["hist"] += hist_values
    return hist_dict


def sqlite_get_min_max(path, column, table):
    min_query = "select min({}) from '{}'".format(column, table)
    max_query = "select max({}) from '{}'".format(column, table)
    with sqlite3.connect(str(path)) as con:
        cur = con.cursor()
        cur.execute(min_query)
        min_value = np.array(cur.fetchone())[0]
        cur.execute(max_query)
        max_value = np.array(cur.fetchone())[0]
    return min_value, max_value


def sqlite_to_numpy_generator_join(
    path, left_keys, right_keys, left_table, right_table, n=100
):
    if isinstance(left_keys, str):
        left_keys = [left_keys]
        left_keys = [left_table + "." + key for key in left_keys]
    if isinstance(right_keys, str):
        right_keys = [right_keys]
        right_keys = [right_table + "." + key for key in right_keys]
    query = "select {}, {} from '{}' inner join '{}' on '{}'.event_no = '{}'.event_no".format(
        ", ".join([key for key in left_keys]),
        ", ".join([key for key in right_keys]),
        left_table,
        right_table,
        left_table,
        right_table,
    )
    with sqlite3.connect(str(path)) as con:
        cur = con.cursor()
        cur.execute(query)
        while True:
            results = cur.fetchmany(n)
            if not results:
                break
            else:
                yield np.array(results)
