import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, 
    geom_point, geom_errorbar, geom_density, geom_boxplot,
    scale_x_log10, scale_y_log10, scale_y_discrete, scale_x_discrete,
    xlim, ylim,
    ggtitle
)

# reproducibility: read frozen data
data_path = "bench/freeze/data/sum_benchmarks.csv" 
df = pd.read_csv(data_path)


# aggregates

agg_df = df.groupby(["dim", "method"]).agg({
    'runtime': ['mean', 'median', 'min', 'max', 'std'], 
    'value': ['mean', 'median']
}).reset_index()
agg_df.columns = ['_'.join(col).strip().rstrip("_") for col in agg_df.columns.values]

p1 = (
    ggplot(agg_df, aes(x='dim', y='runtime_mean', fill='method'))
    + geom_point(size=4)
    + scale_y_log10()
    + scale_x_log10()
    + ggtitle("Mean, Min & Max Runtime per Method")
    + geom_errorbar(
        aes(ymin="runtime_min", ymax="runtime_max"),
        width=0.1
    )
)
# p1.show()
# p1.save("bench/mean_runtime.png")


# runtime per dim

df["dim_normalized_runtime"] = df.apply(
    axis=1,
    func= lambda row: row['runtime'] / row['dim']
)
p2 = (
    ggplot(df, aes(x="dim", y="dim_normalized_runtime", fill="method"))
    + geom_point(size=2)
    + scale_x_log10()
)
# p2.save("bench/runtime-per-dim.png")

# result difference distribution

# pivot to get results for the same rep relative to kahan
df.rename(axis=1, mapper={"value":"result"}, inplace=True)
rep_results = df.pivot(index=["rep", "dim"], columns="method", values="result").reset_index()
rep_results.index.name = "index"
rep_results["kahan"] = 10 ** 12 * (rep_results["kahan"] - rep_results["armadillo"]) / rep_results["armadillo"]
rep_results["pairwise"] = 10 ** 12 * (rep_results["pairwise"] - rep_results["armadillo"]) / rep_results["armadillo"]
# print(rep_results.head())

# melt back for ggplot
melted_df = (
    rep_results
        .melt(id_vars=["dim", "rep"], value_vars=["kahan", "pairwise"])
)
melted_df=melted_df.rename(
    axis=1,
    mapper={
        "value": "relative_difference", 
    }
)
p3 = (
    ggplot(melted_df)
    # + geom_density(alpha=0.5)
    + geom_boxplot(aes(y="relative_difference", x="method"))
    + scale_x_discrete()
    # + scale_x_log10()
)
# p3.show()

p4 = (
    ggplot(melted_df, aes(x="relative_difference", fill="method"))
    + geom_density(alpha=0.4)
    + xlim(-0.002, 0.002) # avoid outliers that throw scale off
    + ggtitle("Relative result difference x10^12 w/r Armadillo arma::accu")
)
p4.save("bench/difference_distr.png")

