import os
import museval
import seaborn as sns

path_eval = "models"
methods = museval.aggregate.MethodStore()

for root, lst_dir, lst_file in os.walk(path_eval):
    for f in lst_file:
        if f.endswith('.pandas'):
            path_df = os.path.join(root, f)
            print(path_df)
            results = museval.aggregate.EvalStore()
            results.load(path_df)
            print(results)
            methods.add_evalstore(results, name=path_df)

df = methods.df.groupby(
    ['method', 'track', 'target', 'metric']
).median().reset_index()

# Get sorting keys (sorted by median of SDR:vocals)
df_sort_by = df[
    (df.metric == "SDR") &
    (df.target == "vocals")
]

# sort methods by score
methods_by_sdr = df_sort_by.score.groupby(
    df_sort_by.method
).median().sort_values().index.tolist()
      
g = sns.FacetGrid(
    df, row="target", col="metric",
    row_order=['vocals', 'drums', 'bass', 'other'],
    height=5, sharex=False, aspect=0.8
)
g = (g.map(
    sns.boxplot,
    "score", 
    "method",
    orient='h',
    order=methods_by_sdr[::-1],
    showfliers=False,
    notch=True
).add_legend())

g.savefig(path_eval + "/boxplot.pdf")
