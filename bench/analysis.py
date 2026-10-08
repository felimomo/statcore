import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, 
    geom_point, geom_errorbar,
    scale_x_log10, scale_y_log10,
    ggtitle
)

# reproducibility: read frozen data
data_path = "bench/freeze/data/sum_benchmarks.csv" 
df = pd.read_csv(data_path)

agg_df = df.groupby(["dim", "method"]).agg({
    'runtime': ['mean', 'median', 'min', 'max', 'std'], 
    'value': ['mean', 'median']
}).reset_index()
agg_df.columns = ['_'.join(col).strip().rstrip("_") for col in agg_df.columns.values]

print(agg_df.columns)
print(agg_df.head())

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
p1.save("bench/mean_runtime.png")


df["dim_normalized_runtime"] = df.apply(
    axis=1,
    func= lambda row: row['runtime'] / row['dim']
)
p2 = (
    ggplot(df, aes(x="dim", y="dim_normalized_runtime", fill="method"))
    + geom_point(size=2)
    + scale_x_log10()
)
p2.save("bench/runtime-per-dim.png")
