import pandas as pd
import scipy.special
from nltk.metrics.distance import edit_distance
from sklearn.metrics import accuracy_score
import numpy as np
import csv
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="chatglm6b-dddd")
parser.add_argument(
    "--systems",
    type=str,
    default="Mac,Android,Thunderbird,HealthApp,OpenStack,OpenSSH,Proxifier,HPC,Zookeeper,Hadoop,Linux,HDFS,BGL,Windows,Apache,Spark",
)
parser.add_argument("--train_percentage", type=str, default="0.025")
args = parser.parse_args()

project_list = args.systems.split(",")


def evaluate(df_parsedlog):
    df_parsedlog["Predict_NoSpaces"] = df_parsedlog["Predict"].str.replace(
        "\s+", "", regex=True
    )
    df_parsedlog["EventTemplate_NoSpaces"] = df_parsedlog["EventTemplate"].str.replace(
        "\s+", "", regex=True
    )
    accuracy_exact_string_matching = accuracy_score(
        np.array(df_parsedlog.EventTemplate_NoSpaces.values, dtype="str"),
        np.array(df_parsedlog.Predict_NoSpaces.values, dtype="str"),
    )
    edit_distance_result = []
    for i, j in zip(
        np.array(df_parsedlog.EventTemplate_NoSpaces.values, dtype="str"),
        np.array(df_parsedlog.Predict_NoSpaces.values, dtype="str"),
    ):
        edit_distance_result.append(edit_distance(i, j))

    edit_distance_result_mean = np.mean(edit_distance_result)
    edit_distance_result_std = np.std(edit_distance_result)
    (precision, recall, f_measure, accuracy_GA) = get_accuracy(
        df_parsedlog["EventTemplate_NoSpaces"], df_parsedlog["Predict_NoSpaces"]
    )
    return (
        accuracy_GA,
        accuracy_exact_string_matching,
        edit_distance_result_mean,
        edit_distance_result_std,
    )


def get_accuracy(series_groundtruth, series_parsedlog, debug=False):
    series_groundtruth_valuecounts = series_groundtruth.value_counts()
    real_pairs = 0
    for count in series_groundtruth_valuecounts:
        if count > 1:
            real_pairs += scipy.special.comb(count, 2)

    series_parsedlog_valuecounts = series_parsedlog.value_counts()
    parsed_pairs = 0
    for count in series_parsedlog_valuecounts:
        if count > 1:
            parsed_pairs += scipy.special.comb(count, 2)

    accurate_pairs = 0
    accurate_events = 0  # determine how many lines are correctly parsed
    for parsed_eventId in series_parsedlog_valuecounts.index:
        logIds = series_parsedlog[series_parsedlog == parsed_eventId].index
        series_groundtruth_logId_valuecounts = series_groundtruth[logIds].value_counts()
        error_eventIds = (
            parsed_eventId,
            series_groundtruth_logId_valuecounts.index.tolist(),
        )
        error = True
        if series_groundtruth_logId_valuecounts.size == 1:
            groundtruth_eventId = series_groundtruth_logId_valuecounts.index[0]
            if (
                logIds.size
                == series_groundtruth[series_groundtruth == groundtruth_eventId].size
            ):
                accurate_events += logIds.size
                error = False
        if error and debug:
            print(
                "(parsed_eventId, groundtruth_eventId) =",
                error_eventIds,
                "failed",
                logIds.size,
                "messages",
            )
        for count in series_groundtruth_logId_valuecounts:
            if count > 1:
                accurate_pairs += scipy.special.comb(count, 2)

    precision = float(accurate_pairs) / parsed_pairs if parsed_pairs > 0 else 0
    recall = float(accurate_pairs) / real_pairs if real_pairs > 0 else 0
    f_measure = 2 * precision * recall / (precision + recall) if (precision+recall) > 0 else 0
    accuracy = float(accurate_events) / series_groundtruth.size
    return precision, recall, f_measure, accuracy


def seen_template(project, precentage):
    file = "../logs/" + project + "/" + precentage + "/train.json"
    df = pd.read_json(file)
    template_list = df["output"].tolist()
    return template_list


def get_unseen_logs(model_name, project, precentage):
    template_list = seen_template(project, precentage)
    logs = []
    predict_logs = []
    predict_log_template = []
    prediction_file = "../output/{}/{}/{}/prediction.csv".format(
        model_name, project, precentage
    )
    with open(prediction_file) as f:
        f = csv.reader(f)
        for i in f:
            if i[2] not in template_list:
                logs.append(i[0])
                predict_logs.append(i[1])
                predict_log_template.append(i[2])
    df = pd.DataFrame(logs, columns=["log"])
    df["Predict"] = predict_logs
    df["EventTemplate"] = predict_log_template
    return df


def unseen_accuracy(model_name,project_list,precentages):
    for precentage in precentages:
        results=[]
        for project in project_list:
            df=get_unseen_logs(model_name, project, precentage)
            if df.empty:
                print(project, "unseen "+precentage, None, None, None, None)
                results.append([project, None, None, None, None])
            else:
                GA, PA, ED, ED_std = evaluate(df)
                print(project, "unseen "+precentage, GA, PA, ED, ED_std)
                results.append([project, GA, PA, ED, ED_std])
        ls_head = ["project", "GP", "PA", "ED", "ED_std"]
        file_path = "evaluation/"+model_name+"/" + precentage + "/"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        with open(file_path + "unseen_result.csv", "w") as f:
            f = csv.writer(f)
            f.writerow(ls_head)
            f.writerows(results)


model = args.model

results = []
for project in project_list:
    file_path = "../output/{}/{}/{}/prediction.csv".format(args.model, project, args.train_percentage)
    column_names = ["log", "Predict", "EventTemplate"]
    df_parsedlog = pd.read_csv(
        file_path, index_col=False, header=None, names=column_names
    )
    GA, PA, ED, ED_std = evaluate(df_parsedlog)
    print(project, args.train_percentage, GA, PA, ED, ED_std)
    results.append([project, GA, PA, ED, ED_std])

ls_head = ["project", "GP", "PA", "ED", "ED_std"]
file_path = "evaluation/" + model + "/" + args.train_percentage + "/"
if not os.path.exists(file_path):
    os.makedirs(file_path)
with open(file_path + "result.csv", "w") as f:
    f = csv.writer(f)
    f.writerow(ls_head)
    f.writerows(results)

#unseen
# unseen_accuracy(model,project_list,[args.train_percentage])
