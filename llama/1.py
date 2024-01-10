import csv
import os
import numpy as np


percentage="0.05"

PROJECT_LIST = [
                'Mac',
                'Android',
                'Thunderbird',                
                'HealthApp',
                'OpenStack',
                'OpenSSH',
                'Proxifier',
                'HPC',
                'Zookeeper',
                'Hadoop',
                'Linux',
                'HDFS',
                'BGL',
                'Windows',
                'Apache',
                'Spark']

def keep_first_line(text):
    lines = text.split("\n")
    first_line = lines[0]
    return first_line

def accuracy(string1, string2):
    # 去除空格
    string1 = string1.replace(" ", "")
    string2 = string2.replace(" ", "")

    # 比较字符串是否相等
    if string1 == string2:
        return True
    else:
        return False


def get_accruacy_over_list(prediction, groundtruth):
    if type(groundtruth) == list:
        if len(groundtruth) == 0:
            return 0
        return np.max([accuracy(prediction, gt) for gt in groundtruth])
    return accuracy(prediction, groundtruth)


def acc(predictions, data, metric="ACC"):
    if metric == "ACC":
        accs = []
        for prediction, dp in zip(predictions, data):
            accs.append(get_accruacy_over_list(prediction, dp))
        return np.mean(accs)

for project in PROJECT_LIST:
    source="./diff/"+project+"/"+percentage+"/"
    out_dir="./diff1/"+project+"/"+percentage+"/"
    inputs=[]
    predictions=[]
    gt=[]
    with open(source+"prediction.csv") as f1:
        f1=csv.reader(f1)
        for i in f1:
            inputs.append(i[0])
            predictions.append(keep_first_line(i[1]))
            gt.append(i[2])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(out_dir + "prediction.csv", "w") as f:
        f = csv.writer(f)
        for i in range(len(predictions)):
            f.writerow([inputs[i], predictions[i], gt[i]])
    with open(out_dir + "diff.csv", "w") as f:
        f = csv.writer(f)
        for i in range(len(predictions)):
            if not accuracy(predictions[i], gt[i]):
                f.writerow([inputs[i], predictions[i], gt[i]])
    score = acc(predictions, gt)
    with open(out_dir + "result.txt", "w") as f:
        f.write(str(score))