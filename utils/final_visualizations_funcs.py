import pandas as pd
import plotly.graph_objects as go

def train_test_all_box(model_names=['GP', 'GSGP', 'SLIM-GSGP', 'NN', 'SLIM-GSGP']):
    fig = go.Figure()
    
    BOX_WIDTH = 0.25   # Width of individual boxes
    
    # We'll track if we've added legend entries already
    train_legend_added = False
    test_legend_added = False
    
    for i, model_name in enumerate(model_names):
        LOG_PATH = f"./log/{model_name}/{model_name}_sustavianfeed_outer_results.csv"
        df_results = pd.read_csv(LOG_PATH)
        comb_train_values = df_results['rmse_train'].values
        comb_test_values = df_results['rmse_test'].values
        
        # Calculate x positions - closer together within each model group
        train_x = i - 0.15  # Slightly left
        test_x = i + 0.15   # Slightly right
        
        # Add train boxplot (only show legend for first occurrence)
        fig.add_trace(
            go.Box(
                y=comb_train_values,
                x=[train_x] * len(comb_train_values),
                width=BOX_WIDTH,
                boxpoints="all",
                jitter=0.3,
                pointpos=0,
                line=dict(color="orange", width=1),
                fillcolor='rgba(255,165,0,0.5)',
                name="Train" if not train_legend_added else "",
                legendgroup="train",
                showlegend=not train_legend_added
            )
        )
        train_legend_added = True
        
        # Add test boxplot (only show legend for first occurrence)
        fig.add_trace(
            go.Box(
                y=comb_test_values,
                x=[test_x] * len(comb_test_values),
                width=BOX_WIDTH,
                boxpoints="all",
                jitter=0.3,
                pointpos=0,
                line=dict(color="blue", width=1),
                fillcolor='rgba(0,0,255,0.5)',
                name="Test" if not test_legend_added else "",
                legendgroup="test",
                showlegend=not test_legend_added
            )
        )
        test_legend_added = True
    
    # Update layout for compact display
    fig.update_layout(
        title_text='Models Comparison (Train vs Test)',
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(model_names))),
            ticktext=model_names,
            title='Models',
            range=[-0.5, len(model_names)-0.5]
        ),
        yaxis=dict(title='RMSE'),
        boxmode='group',
        boxgap=0.3,
        boxgroupgap=0.2,
        margin=dict(l=50, r=50, t=80, b=50),
        template="plotly_white",
        width=800,
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.show()


# auxiliar function to convert hex to rgb
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def evolution_plots_all(k_outer:int, model_names: list, show_train:bool=True, show_test:bool=True):
    fig= go.Figure()

    colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b']
    
    for i, model in enumerate(model_names):
        LOG_DIR = "./log/" + model + "/" + model + "_sustavianfeed"
        df_log = []  # group all outers here
        for i_outer in range(k_outer):
            df = pd.read_csv(LOG_DIR + f"_outer_{i_outer}.csv", header=None)
            df["cv"] = i_outer
            df_log.append(df)
        df_log = pd.concat(df_log, ignore_index=True)
        
        df_plot = pd.DataFrame(
                {
                    "x": df_log.iloc[:, 4],
                    "rmse": abs(df_log.iloc[:, 5]),
                    "rmse_val": abs(df_log.iloc[:, 8]),
                    "size": df_log.iloc[:, 9],
                }
            )
        agg = df_plot.groupby("x")['rmse'].agg(["mean", "std"]).reset_index()
        agg["y_upper"] = agg["mean"] + agg["std"]
        agg["y_lower"] = agg["mean"] - agg["std"]
        agg.loc[agg["y_lower"] < 0, "y_lower"] = 0

        color = colors[i]

        if show_train:
            fig.add_trace(
                go.Scatter(
                    x=agg["x"],
                    y=agg["mean"],
                    mode="lines",
                    name=f"{model} (Train)",
                    line=dict(color=color, dash='dash'),
                    legendgroup=model,  # Group traces by model
                    showlegend=True,
                ),
            )
            fig.add_trace(
                go.Scatter(
                    x=agg["x"],
                    y=agg["y_upper"],
                    mode="lines",
                    name="+1 std Train",
                    line=dict(width=0),
                    legendgroup=model,
                    showlegend=False,
                ),
            )
            fig.add_trace(
                go.Scatter(
                    x=agg["x"],
                    y=agg["y_lower"],
                    mode="lines",
                    name="-1 std Train",
                    fill="tonexty",
                    fillcolor=f"rgba{(*hex_to_rgb(color), 0.1)}",
                    line=dict(width=0),
                    legendgroup=model,
                    showlegend=False,
                ),
            )

        if show_test:
            agg = df_plot.groupby("x")["rmse_val"].agg(["mean", "std"]).reset_index()
            agg["y_upper"] = agg["mean"] + agg["std"]
            agg["y_lower"] = agg["mean"] - agg["std"]

            fig.add_trace(
                go.Scatter(
                    x=agg["x"],
                    y=agg["mean"],
                    mode="lines",
                    name=f"{model} (Test)",
                    line=dict(color=color),
                    legendgroup=model,
                    showlegend=True,
                ),
            )
            fig.add_trace(
                go.Scatter(
                    x=agg["x"],
                    y=agg["y_upper"],
                    mode="lines",
                    name="+1 std Val",
                    line=dict(width=0),
                    legendgroup=model,
                    showlegend=False,
                ),
            )
            fig.add_trace(
                go.Scatter(
                    x=agg["x"],
                    y=agg["y_lower"],
                    mode="lines",
                    name="-1 std Val",
                    fill="tonexty",
                    fillcolor=f"rgba{(*hex_to_rgb(color), 0.1)}",
                    line=dict(width=0),
                    legendgroup=model,
                    showlegend=False,
                ),
            )


    fig.update_layout(
        title_text='Evolution of RMSE (Comparison between Models)',
        xaxis_title="Iterations",
        yaxis_title="RMSE",
        height=500,
        width=800,
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=-0.50, 
            xanchor="center", 
            x=0.5,
            #traceorder="normal",  # Keep related items together
            #itemsizing="constant"  # Keep legend item sizes consistent
        ),
    )
    fig.update_yaxes(range=[0, None])
    fig.show()
