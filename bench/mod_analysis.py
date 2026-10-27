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
    scale_color_gradient,
    scale_color_continuous,
    labs,
    theme,
    element_rect,
    element_line,
)
import pathlib

data_loc = pathlib.Path("bench/data")
accu_df = pd.read_csv(data_loc / "Gaussian_GLM_accu_bmk.csv")
# print(accu_df.head())

max_err_df = accu_df.groupby(['dim', 'N']).agg(
    max_err_mean = ('max_err', 'mean'),
    max_err_std = ('max_err', 'std')
).reset_index()

max_err_df['dim_to_N'] = max_err_df['dim'] / max_err_df['N']

print(max_err_df.head(10))

p = (
    ggplot(max_err_df, aes(x = "dim", y = "max_err_mean", color = "dim_to_N"))
    + geom_point(size=2)
    + scale_x_log10()
    + scale_y_log10()
    + scale_color_continuous(cmap_name="viridis", trans="log10", name="Inner-dim.\n" "to Data-N Ratio")
    + geom_errorbar(aes(ymin="max_err_mean-max_err_std", ymax="max_err_mean+max_err_std", width=0.02))
    + labs(y="Max Error", x="Inner Dimension", title="Max Error in Gauss-Linear GLM fit (N_rep=25)")
    + theme(
        panel_background=element_rect(fill="#FFFEFE"),  
        panel_border=element_rect(color='black', size=1, fill=None),
        axis_ticks=element_line(color="black", size=1),
        axis_ticks_length=4,
        panel_grid_major=element_line(
            color="#bdbdbd", size=0.5, linetype="solid"
        ),
        # panel_grid_minor=element_line(
        #     color="#bdbdbd", size=0.5, linetype="solid"
        # ),
    )
)
p.show()
