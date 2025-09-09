import pandas as pd
import numpy as np
from functools import reduce
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score
import scipy.stats as stats
import os

# -------- helpers --------
def process_name(name):
    return name if name == "label" else name + "_label"

def confidence_interval(series, confidence=0.95):
    # series: 1D numeric pandas Series
    x = pd.to_numeric(series, errors="coerce").dropna()
    n = len(x)
    mean = x.mean()
    std_err = x.std(ddof=1) / np.sqrt(n)
    margin = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
    return (mean - margin, mean + margin)

def pretty_model(name: str) -> str:
    # Optional: prettify model names for plots
    return (name
            .replace("_", " ")
            .replace("inform", "InForm")
            .replace("cellprofiler", "CellProfiler")
            .replace("cellpose", "Cellpose")
            .replace("spatial metrics", "Spatial Metrics")
            .title())

# -------- data --------
# Put human readers here
pathologist_names = ["patrick", "niclos", "fredrik_heldin", "gordan"]

# All evaluators (humans + models)
names = [
    #"patrick", "niclos", "fredrik_heldin", "gordan",
    "cellprofiler_spatial_metrics_model",
    "cellpose_spatial_metrics_model",
    "inform_spatial_metrics_model",
]

# Build anonymized display names for plots ONLY
pathologist_map = {
    p: f"Pathologist {i+1}" for i, p in enumerate(pathologist_names)
}
display_names = [pathologist_map.get(n, pretty_model(n)) for n in names]

pathologists_datas = []
for name in names:
    data = pd.read_csv(os.path.join("subtype_annotator_labels", name + "_labels.csv"))
    data.rename(columns={"Patient ID": "ID"}, inplace=True)
    data[name + "_label"] = data["Label"].map(lambda x: 1 if x == "Adeno carcinoma" else 0)
    data = data[["ID", name + "_label"]]
    pathologists_datas.append(data)

pathologists_df = reduce(lambda left, right: pd.merge(left, right, on="ID", how="outer"), pathologists_datas)

test_data = pd.read_csv("../multiplex_dataset/lung_cancer_BOMI2_dataset/binary_subtype_prediction_ACvsSqCC/static_split/test.csv")
test_data = pd.merge(test_data[["ID", "label"]], pathologists_df, how="outer")

# -------- metrics table --------
stats_dict = {"accuracy": [], "precision": [], "recall": []}
for name in names:
    y_true = test_data["label"]
    y_pred = test_data[process_name(name)]
    accuracy = (y_true == y_pred).mean()
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    stats_dict["accuracy"].append(accuracy)
    stats_dict["precision"].append(precision)
    stats_dict["recall"].append(recall)
    print(name, "accuracy:", accuracy)

stats_df = pd.DataFrame(stats_dict, index=names)
print("mean accuracy: ", stats_df["accuracy"].mean())
print("mean precision:", stats_df["precision"].mean())
print("mean recall:   ", stats_df["recall"].mean())
print(stats_df)

# -------- kappa heatmap (anonymized labels) --------
kappas = np.ones((len(names)+1, len(names)+1))
for i, name1 in enumerate(["label"] + names):
    for j, name2 in enumerate(["label"] + names):
        kappas[i, j] = cohen_kappa_score(test_data[process_name(name1)],
                                         test_data[process_name(name2)])

kappa_df = pd.DataFrame(
    kappas,
    index=["Ground truth"] + display_names,   # anonymized y labels
    columns=["Ground truth"] + display_names  # anonymized x labels
)

plt.figure()
sns.heatmap(kappa_df, annot=True, cmap="Blues", vmin=0, vmax=1)
plt.title("Cohen's Kappa Agreement")
plt.tight_layout()

# -------- per-case agreement heatmap (anonymized row labels) --------
test_data["correct predictions"] = (test_data.iloc[:, 2:] == test_data["label"].values[:, None]).sum(axis=1)
df = test_data
df_sorted = (
    df.sort_values(by=["label", "correct predictions"], ascending=[False, False])
    .reset_index(drop=True)
)

# Ensure SqCC (0) block sorted with fewest correct at top (as you had)
df_sorted.loc[df_sorted["label"] == 0] = (
    df_sorted[df_sorted["label"] == 0]
    .sort_values(by="correct predictions", ascending=True)
    .values
)

df_sorted.index = df_sorted["ID"]
print(df_sorted)

plt.figure()
# Remove "_label" in column headers only for plotting
plot_cols = ["label"] + [c for c in df_sorted.columns if c.endswith("_label")]
plot_mat = df_sorted[plot_cols].copy()
plot_mat.columns = ["label"] + display_names  # anonymized row labels

# We want evaluators on rows -> transpose
ax = sns.heatmap(
    plot_mat.transpose(),
    annot=True, cmap="Blues", vmin=0, vmax=1,
    yticklabels=["Ground truth"] + display_names  # enforce anonymized labels
)

# Force all x labels (case IDs) to show
ax.set_xticks(range(len(df_sorted)))
ax.set_xticklabels(df_sorted.index, rotation=90, ha="right")
plt.title("Per-case Labels (1 = Adeno, 0 = SqCC)")
plt.tight_layout()
plt.show()
