# %%
import csv
import os
import pathlib
import re

import pandas as pd

pd.set_option("display.precision", 4)
pathlib.Path("results").mkdir(exist_ok=True)

# %%
dir = "../run/"
dir = pathlib.Path(dir)

epoch_regex = r"Epoch (\d+)\/(\d+), Acc=([\d.]+), Val Loss=([\d.]+)"

results = []
for dataset_dir in dir.iterdir():
    dataset_dir = dataset_dir / "prune"

    for model_dir in dataset_dir.iterdir():
        splits = str(model_dir).split("-")
        dataset = splits[0].split("/")[-1]
        scope = splits[1]
        method = splits[2]
        ratio = splits[3]
        model = splits[4]

        logfile = model_dir / f"{model_dir.name}.txt"
        with open(logfile, "r") as f:
            for line in f:
                match = re.search(epoch_regex, line)
                if match:
                    epoch = match.group(1)
                    total_epoch = match.group(2)
                    accuracy = match.group(3)
                    val_loss = match.group(4)

                    results.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "scope": scope,
                            "ratio": ratio,
                            "method": method,
                            "epoch": epoch,
                            "total_epoch": total_epoch,
                            "accuracy": accuracy,
                            "validation_loss": val_loss,
                        }
                    )

df = pd.DataFrame(results)
df.to_csv("results/raw.csv", index=False)

# %%
df = pd.read_csv("results/raw.csv")
results = []

for dataset in df["dataset"].unique():
    c_dataset = df["dataset"] == dataset

    for model in df["model"].unique():
        c_model = df["model"] == model

        for ratio in df["ratio"].unique():
            c_ratio = df["ratio"] == ratio

            for method in df["method"].unique():
                c_method = df["method"] == method

                exp = df[c_dataset & c_model & c_ratio & c_method]
                if len(exp) == 0:
                    continue

                best_acc = exp["accuracy"].max()
                results.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "ratio": ratio,
                        "method": method,
                        "best_accuracy": best_acc,
                    }
                )

table = pd.DataFrame(results)
table.to_csv("results/clean.csv", index=False)

# %%
df = pd.read_csv("results/clean.csv")
df = df[df["method"] != "proj"]
df["method"] = pd.Categorical(
    df["method"],
    [
        "random",
        "l2",
        "slim",
        "fpgm",
        "obdc",
        "lamp",
        "group_norm",
        "group_sl",
        "proj",
        "proj_sl",
    ],
).rename_categories(
    {
        "random": "Random",  # 0
        "l2": "MagnitudeL2",  # 1
        "slim": "Slimming",  # 2017
        "fpgm": "FPGM",  # 2019
        "obdc": "EigenDamage",  # 2019
        "lamp": "LAMP",  # 2020
        "group_norm": "DepGraph",  # 2023
        "group_sl": "DepGraph-SL",  # 2023
        "proj_sl": "\\bf{Projective (ours)}",
    }
)

for dataset in df["dataset"].unique():
    c_dataset = df["dataset"] == dataset

    for model in df["model"].unique():
        c_model = df["model"] == model

        ratios = []
        skip = False
        for ratio in [2.0, 3.0]:
            c_ratio = df["ratio"] == ratio

            tab = df[c_dataset & c_model & c_ratio]
            if len(tab) == 0:
                skip = True
                break

            tab = tab.rename({"best_accuracy": ratio}, axis=1)
            tab.drop(["ratio"], axis=1, inplace=True)
            tab.set_index("method", inplace=True)

            ratios.append(tab)

        if skip:
            continue

        tab = pd.merge(ratios[0], ratios[1], on=["method", "dataset", "model"])
        tab.sort_index(inplace=True)

        tab[3.0] = tab[3.0].map("{:.4f}".format)

        file = f"results/tables/{dataset}/{model}.csv"
        pathlib.Path(file).parent.mkdir(exist_ok=True, parents=True)
        tab.to_csv(file)


# %%
def transform_emph(df):
    for col in df.columns:
        if col == "method":
            continue

        max1 = df[col].max()
        max2 = df[col][df[col] < max1].max()

        def t(x):
            if x == max1:
                return f"\\bf{{{x:.4f}}}"
            if x == max2:
                return f"\\un{{{x:.4f}}}"
            return f"{x:.4f}"

        df[col] = df[col].apply(t)
    return df


def tex_post(file):
    with open(file, "r") as f:
        s = f.read()
        s = s.replace("&", " & ")
        s = s.replace("\n", " \\\\\n")
    with open(file, "w") as f:
        f.write(s)


# %%
tables = []
for dataset in ["cifar10", "cifar100"]:
    tabs = []
    for model in ["vgg19", "resnet56"]:
        file = f"results/tables/{dataset}/{model}.csv"

        tab = pd.read_csv(file)
        tab.set_index("method", inplace=True)
        tab.drop(["dataset", "model"], axis=1, inplace=True)
        tab.rename(
            {
                "2.0": f"{dataset}-{model}-2.0",
                "3.0": f"{dataset}-{model}-3.0",
            },
            axis=1,
            inplace=True,
        )
        tabs.append(tab)

    tab = pd.merge(tabs[0], tabs[1], on=["method"])
    tables.append(tab)

tables = pd.merge(tables[0], tables[1], on=["method"])
tables = transform_emph(tables)
os.makedirs("results/summary", exist_ok=True)
tables.to_csv("results/summary/vgg-resnet.csv")
tables.to_csv(
    "results/summary/vgg-resnet.tex",
    sep="&",
    header=False,
    quoting=csv.QUOTE_NONE,
)
tex_post("results/summary/vgg-resnet.tex")

# %%
tables = []
for dataset in ["cifar10", "cifar100"]:
    tabs = []
    for model in ["mobilenetv2"]:
        file = f"results/tables/{dataset}/{model}.csv"

        tab = pd.read_csv(file)
        tab.set_index("method", inplace=True)
        tab.drop(["dataset", "model"], axis=1, inplace=True)
        tab.rename(
            {
                "2.0": f"{dataset}-{model}-2.0",
                "3.0": f"{dataset}-{model}-3.0",
            },
            axis=1,
            inplace=True,
        )
        tabs.append(tab)
    tables.append(tabs[0])

for dataset in ["modelnet40"]:
    tabs = []
    for model in ["pointnet"]:
        file = f"results/tables/{dataset}/{model}.csv"

        tab = pd.read_csv(file)
        tab.set_index("method", inplace=True)
        tab.drop(["dataset", "model"], axis=1, inplace=True)
        tab.rename(
            {
                "2.0": f"{dataset}-{model}-2.0",
                "3.0": f"{dataset}-{model}-3.0",
            },
            axis=1,
            inplace=True,
        )
        tabs.append(tab)
    tables.append(tabs[0])


result = pd.merge(tables[0], tables[1], on=["method"])
result = pd.merge(result, tables[2], on=["method"])

result = transform_emph(result)
os.makedirs("results/summary", exist_ok=True)
result.to_csv("results/summary/mobile-point.csv", float_format="%.4f")
result.to_csv(
    "results/summary/mobile-point.tex",
    sep="&",
    header=False,
    quoting=csv.QUOTE_NONE,
)
tex_post("results/summary/mobile-point.tex")

# %%


# %%
all_datasets = [["cifar10", "cifar100"], ["modelnet40"]]
all_models = [["vgg19", "resnet56", "mobilenetv2"], ["pointnet"]]

for datasets, models in zip(all_datasets, all_models):
    for dataset in datasets:
        tabs = []
        for model in models:
            file = f"results/tables/{dataset}/{model}.csv"

            tab = pd.read_csv(file)
            tab.set_index("method", inplace=True)
            tab.drop(["dataset", "model"], axis=1, inplace=True)
            tab.rename(
                {
                    "2.0": f"{dataset}-{model}-2.0",
                    "3.0": f"{dataset}-{model}-3.0",
                },
                axis=1,
                inplace=True,
            )
            tabs.append(tab)

        tab = pd.concat(tabs, axis=1)
        tab = transform_emph(tab)

        file = f"results/tex/{dataset}.tex"
        os.makedirs(os.path.dirname(file), exist_ok=True)
        tab.to_csv(file, sep="&", header=False, quoting=csv.QUOTE_NONE)
        tex_post(file)

# %%
