import pandas as pd
import numpy as np
from math import ceil
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import pickle

"log csvs structure"
# 0  - Algorithm
# 1  - Instance ID
# 2  - Dataset
# 3  - Seed
# 4  - Generation
# 5  - Fitness
# 6  - Running time
# 7  - Population nodes
# 8  - Test fitness
# 9  - Elite nodes
# 10 - niche entropy
# 11 - sd(pop.fit)
# 12 - Log level
# 13 - The parameter combination string (combination description)


def get_comb_values(model_name):
    LOG_PATH = (
        "./log/"
        + model_name
        + "/"
        + model_name
        + "_sustavianfeed"
        + "_outer_results.csv"
    )
    df_results = pd.read_csv(LOG_PATH)
    comb_train_values = {}
    comb_test_values = {}
    for comb in df_results["dynamic_params"].unique():
        df_one_comb = df_results[df_results["dynamic_params"] == comb]
        train_values = df_one_comb["rmse_train"]
        test_values = df_one_comb["rmse_test"]
        comb_train_values[comb] = list(train_values)
        comb_test_values[comb] = list(test_values)

    return comb_train_values, comb_test_values


def test_best_combs(model_name):
    test_rmse_by_config = get_comb_values(model_name=model_name)[1]
    fig = go.Figure()
    for config, rmse_values in test_rmse_by_config.items():
        config_paragraphed = ",<br>".join([part.strip() for part in config.split(",")])
        fig.add_trace(
            go.Box(
                y=rmse_values,
                boxpoints="all",
                jitter=0.5,
                pointpos=0,
                line=dict(color="orange"),
                name=config_paragraphed,
            )
        )

    fig.update_layout(
        title=model_name + " - Best Combinations",
        xaxis_title="",
        yaxis_title="Test RMSE",
        height=500,
        width=1100,
        yaxis_range=[0, None],
        margin=dict(l=50, r=50, t=50, b=20),
        showlegend=False,
        template="plotly_white",
    )

    fig.show()


def train_test_best_combs(model_name):
    train_rmse_by_config = get_comb_values(model_name=model_name)[0]
    test_rmse_by_config = get_comb_values(model_name=model_name)[1]

    fig = make_subplots(
        rows=len(test_rmse_by_config.keys()),
        cols=1,
        subplot_titles=[f"Combination:{i}" for i in test_rmse_by_config.keys()],
    )
    for i, config in enumerate(test_rmse_by_config.keys()):
        config_paragraphed = ",<br>".join([part.strip() for part in config.split(",")])
        fig.add_trace(
            go.Box(
                y=train_rmse_by_config[config],
                boxpoints="all",
                jitter=0.5,
                pointpos=0,
                line=dict(color="orange"),
                name="Train",
            ),
            row=i + 1,
            col=1,
        )

        fig.add_trace(
            go.Box(
                y=test_rmse_by_config[config],
                boxpoints="all",
                jitter=0.5,
                pointpos=0,
                line=dict(color="blue"),
                name="Test",
            ),
            row=i + 1,
            col=1,
        )

    fig.update_layout(
        title_text=model_name + " - Best Combinations",
        height=300
        * len(test_rmse_by_config.keys()),  # Dynamic height based on number of subplots
        width=1100,
        margin=dict(l=50, r=50, t=100, b=50),  # Adjust margins
        template="plotly_white",
        showlegend=False,  # Show legend if needed
    )

    # Update y-axis properties for all subplots
    fig.update_yaxes(title_text="RMSE", range=[0, None], showgrid=True)

    # Optionally adjust title positions
    fig.update_annotations(yshift=20)  # Move subplot titles up slightly

    fig.show()


def fit_and_size_per_outer(k_outer, model_name):
    LOG_DIR = "./log/" + model_name + "/" + model_name + "_sustavianfeed"
    for i_outer in range(k_outer):
        LOG_PATH = LOG_DIR + f"_outer_{i_outer}.csv"
        df = pd.read_csv(LOG_PATH, header=None)
        param_str = df[13][0]

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(f"Fitness evolution", f"Size evolution"),
            horizontal_spacing=0.15,
        )

        fig.add_trace(
            go.Scatter(
                y=df.iloc[:, 5].values,
                mode="lines",
                name="Train",
                line=dict(color="blue"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                y=df.iloc[:, 8].values,
                mode="lines",
                name="Test",
                line=dict(color="orange"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(y=df.iloc[:, 9].values, mode="lines", name="Size"), row=1, col=2
        )
        fig.update_layout(
            width=1600,
            height=400,
            showlegend=True,
            yaxis_range=[0, None],
            legend=dict(
                orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5
            ),
            title_text=f"{model_name} - Outer Fold {i_outer} (Combination: {param_str})",
            title_x=0.5,  # Center title
            title_font=dict(size=18),  # Bigger font
            margin=dict(t=100),
        )

        fig.update_yaxes(title_text="RMSE", row=1, col=1)  # First y-axis title

        fig.update_yaxes(
            title_text="Size",
            type="log",  # Logarithmic scale to handle the large numbers better
            tickformat=".1e",  # Scientific notation with 2 decimal places
            exponentformat="power",  # Shows as ×10ⁿ instead of en
            row=1,
            col=2,
        )

        fig.show()


def make_evolution_plots(
    n_rows,
    n_cols,
    slim_versions,
    df_log,
    plot_title,
    var="rmse",
    model_name="SLIM-GSGP",
    height=None,
    width=None,
):

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"{i}" for i in slim_versions],
        vertical_spacing=0.2,
        horizontal_spacing=0.2,
    )

    for i, sv in enumerate(slim_versions):
        row = i // n_cols + 1
        col = i % n_cols + 1
        show_legend = i == 0

        # Plot data
        df_plot = pd.DataFrame(
            {
                "x": df_log[df_log[13] == sv].iloc[:, 4],
                "rmse": abs(df_log[df_log[13] == sv].iloc[:, 5]),
                "rmse_val": abs(df_log[df_log[13] == sv].iloc[:, 8]),
                "size": df_log[df_log[13] == sv].iloc[:, 9],
            }
        )
        agg = df_plot.groupby("x")[var].agg(["mean", "std"]).reset_index()
        agg["y_upper"] = agg["mean"] + agg["std"]
        agg["y_lower"] = agg["mean"] - agg["std"]
        agg.loc[agg["y_lower"] < 0, "y_lower"] = 0

        fig.add_trace(
            go.Scatter(
                x=agg["x"],
                y=agg["mean"],
                mode="lines",
                name="Train" if var == "rmse" else "Size",
                line=dict(color="blue"),
                showlegend=show_legend,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=agg["x"],
                y=agg["y_upper"],
                mode="lines",
                name="+1 std Train",
                line=dict(width=0),
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=agg["x"],
                y=agg["y_lower"],
                mode="lines",
                name="-1 std Train",
                fill="tonexty",
                fillcolor="rgba(0,0,255,0.1)",
                line=dict(width=0),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        if var == "rmse":
            agg = df_plot.groupby("x")["rmse_val"].agg(["mean", "std"]).reset_index()
            agg["y_upper"] = agg["mean"] + agg["std"]
            agg["y_lower"] = agg["mean"] - agg["std"]
            fig.add_trace(
                go.Scatter(
                    x=agg["x"],
                    y=agg["mean"],
                    mode="lines",
                    name="Test",
                    line=dict(color="orange"),
                    showlegend=show_legend,
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=agg["x"],
                    y=agg["y_upper"],
                    mode="lines",
                    name="+1 std Val",
                    line=dict(width=0),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=agg["x"],
                    y=agg["y_lower"],
                    mode="lines",
                    name="-1 std Val",
                    fill="tonexty",
                    fillcolor="rgba(255,165,0,0.1)",
                    line=dict(width=0),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

    fig.update_layout(
        title_text=plot_title,
        xaxis_title="Generations" if model_name != "NN" else "Epochs",
        yaxis_title="RMSE",
        height=1000 if not height else height,
        width=1600 if not width else width,
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5
        ),
    )
    fig.update_yaxes(range=[0, None])
    fig.show()


def fit_or_size_per_comb(k_outer, model_name, size=False, height=None, weight=None):
    LOG_DIR = "./log/" + model_name + "/" + model_name + "_sustavianfeed"
    df_log = []  # group all outers here
    comb_list = []  # log all unique combinations
    for i_outer in range(k_outer):
        df = pd.read_csv(LOG_DIR + f"_outer_{i_outer}.csv", header=None)
        df["cv"] = i_outer
        param_str = df[13][0]
        comb_list.append(param_str)
        df_log.append(df)
    df_log = pd.concat(df_log, ignore_index=True)

    unique_comb_list = list(set(comb_list))
    n_combinations = len(unique_comb_list)
    # deciding on the number of cols and rows
    n_cols = ceil(n_combinations**0.3)
    n_rows = ceil(n_combinations / n_cols)

    if not size:
        make_evolution_plots(
            n_rows=n_rows,
            n_cols=n_cols,
            slim_versions=unique_comb_list,
            df_log=df_log,
            plot_title=f"{model_name} - Train vs Test Fitness",
            model_name=model_name,
            height=height,
            weight=weight
        )

    if size:
        make_evolution_plots(
            n_rows=n_rows,
            n_cols=n_cols,
            slim_versions=unique_comb_list,
            df_log=df_log,
            var="size",
            plot_title="Size (" + model_name + " dataset)",
            model_name=model_name,
            height=height,
            weight=weight
        )


"""

For the next delivery


"""


def niche_entropy(k_outer, model_name, skip_n_gens: int = None):
    LOG_DIR = "./log/" + model_name + "/" + model_name + "_sustavianfeed"

    # deciding on the number of cols and rows
    cols = ceil(k_outer**0.3)
    rows = ceil(k_outer / cols)

    fig = sp.make_subplots(
        rows=rows, cols=cols, subplot_titles=[f"Outer Fold {i}" for i in range(k_outer)]
    )

    for i_outer in range(k_outer):
        LOG_PATH = LOG_DIR + f"_outer_{i_outer}.csv"
        df = pd.read_csv(LOG_PATH, header=None)
        param_str = df[13][0]
        if len(param_str) >= 60:  # divide param_str if it is too long
            param_str1 = param_str[:59]
            param_str2 = param_str[59:]
        # skip n gnerations to plot from it on, and to make more visible the later flutuations
        if skip_n_gens:
            df = df.drop(index=df.head(skip_n_gens).index)
        try:
            div_vector_log = df.iloc[:, 10].values
            div_vector_values = np.array(
                [
                    float(x.replace("tensor(", "").replace(")", ""))
                    for x in div_vector_log
                ]
            )
        except:
            try:
                div_vector_values = df.iloc[:, 10].values
            except Exception as e:
                print(e)

        row = (i_outer // cols) + 1
        col = (i_outer % cols) + 1

        fig.add_trace(
            go.Scatter(
                y=div_vector_values,
                mode="lines",
                name="Niche Entropy",
                line=dict(color="orange"),
                showlegend=(i_outer == 0),
            ),
            row=row,
            col=col,
        )

        if len(param_str) >= 60:
            fig.layout.annotations[i_outer].update(
                text=f"Outer Fold {i_outer}<br>{param_str1}<br>{param_str2}",
                font=dict(size=11),
            )
        else:
            fig.layout.annotations[i_outer].update(
                text=f"Outer Fold {i_outer}<br>{param_str}", font=dict(size=11)
            )

    fig.update_layout(
        height=200 * rows,
        width=425 * cols,
        title_text=f"{model_name} - Niche Entropy/Pop Semantic Diversity (x=Generation, y=Entropy)",
        title_y=0.97,
        margin=dict(t=100, l=50, r=50, b=50),
    )

    fig.show()


def pop_fitness_diversity(k_outer, model_name, skip_n_gens: int = None):
    LOG_DIR = "./log/" + model_name + "/" + model_name + "_sustavianfeed"
    for i_outer in range(k_outer):
        LOG_PATH = LOG_DIR + f"_outer_{i_outer}.csv"
        df = pd.read_csv(LOG_PATH, header=None)
        param_str = df[13][0]  # first string
        if skip_n_gens:
            df = df.drop(index=df.head(skip_n_gens).index)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                y=df.iloc[:, 11].values,
                mode="lines",
                name="Train",
                line=dict(color="orange"),
            )
        )
        fig.update_layout(
            height=400,
            width=900,
            yaxis_range=[0, None],
            title_text=f"{model_name} - Population Fitness Diversity<br>(Outer fold {i_outer}: Comb {param_str})",
            title_font=dict(size=15),
            xaxis_title="Generation",
            yaxis_title="Fitness Standard Deviation",
            title_y=0.93,
            margin=dict(t=80, l=50, r=80, b=50),
        )
        fig.show()


def plot_species(k_outer, model_name="NEAT"):
    """Visualizes speciation throughout evolution."""

    for i_outer in range(k_outer):
        LOG_PATH = (
            "./log/"
            + model_name
            + "/"
            + f"{model_name}_sustavianfeed_outer_{i_outer}.csv.pkl"
        )
        with open(LOG_PATH, "rb") as file:
            statistics = pickle.load(file)

        species_sizes = statistics.get_species_sizes()
        num_generations = len(species_sizes)
        curves = np.array(species_sizes).T

        fig, ax = plt.subplots()
        ax.stackplot(range(num_generations), *curves)

        plt.suptitle(f"{model_name} - Outer {i_outer}")
        plt.title("Speciation")
        plt.ylabel("Size per Species")
        plt.xlabel("Generations")

        plt.show()
