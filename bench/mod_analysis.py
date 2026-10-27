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
import pathlib

data_loc = pathlib.Path("bench/data")
accu_df = pd.read_csv(data_loc / "Gaussian_GLM_accu_bmk.csv")
# print(accu_df.head())

max_err_df = accu_df.groupby(['dim', 'N']).agg(
    max_err_mean = ('max_err', 'mean'),
    max_err_std = ('max_err', 'std')
    )

print(max_err_df.head(10))
