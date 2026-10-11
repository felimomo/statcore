import pandas as pd
from plotnine import (
    ggplot, aes, 
    geom_point, geom_density,
    geom_line,
    geom_errorbar,
    geom_jitter,
    scale_x_log10,
    scale_y_log10,
    scale_color_discrete,
    labs,
)

df = pd.read_csv("bench/freeze/data/matmult_benchmarks.csv")

agg_runtimes = df.groupby(["dim", "method"]).agg({"runtime": ["mean", "std"]}).reset_index()
agg_runtimes.columns = ["dim", "method", "mean", "std"] # flatten col names
# print(agg_runtimes.head())

p_simd = (
    ggplot(
        agg_runtimes[agg_runtimes['method'].isin(["simd128", "simd64", "simd32"])], 
        aes(x='dim', y='mean', color='method')
    )
    + geom_line()
    + scale_x_log10()
    + scale_y_log10()
    + geom_errorbar(aes(ymin="mean-std", ymax="mean+std", width=0.02))
    + labs(y="Runtime (s)", x="Dimension", title="Mat. Mult. Runtime per Method")
    + scale_color_discrete(
        labels=[
            "128x128 SIMD-Block Mult.", "32x32 SIMD-Block Mult.", "64x64 SIMD-Block Mult.", 
        ]
    )
)
# p_simd.show()
# p_simd.save("bench/freeze/simd_runtime.png")

p2 = (
    ggplot(
        agg_runtimes[agg_runtimes['method'].isin(["ijk", "ikj", "block32", "block64"])], 
        aes(x='dim', y='mean', color='method')
    )
    # + geom_point(size=4)
    + geom_line()
    + scale_x_log10()
    + scale_y_log10()
    + geom_errorbar(aes(ymin="mean-std", ymax="mean+std", width=0.02))
    + labs(y="Runtime (s)", x="Dimension", title="Mat. Mult. Runtime per Method")
    + scale_color_discrete(
        labels=[
            "32x32 Block Mult.", "64x64 Block Mult.", "naive ijk loop", "ill-optimized ikj loop"
        ]
    )
)
# p2.show()
# p2.save("bench/freeze/non_simd_runtime.png")


err_df = df.groupby(["dim", "method"]).agg({"error": ["mean", "std"]}).reset_index()
err_df.columns = ["dim", "method", "mean_error", "std"] # flatten col names
print(err_df.head())

p3=(
    ggplot(err_df[err_df["method"].isin(["block64", "ijk"])], aes(x='dim', y='mean_error', color='method'))
    + geom_line()
    + scale_y_log10()
    + labs(x="Dimension", y="Error", title="Error relative to Armadillo matrix product")
    + scale_color_discrete(
        labels=[
            "Block Mat. Methods",
            "ijk-like Methods"
        ]
    )
)
p3.show()
p3.save("bench/freeze/errors.png")