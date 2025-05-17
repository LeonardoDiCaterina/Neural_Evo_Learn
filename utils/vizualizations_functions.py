import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def make_evolution_plots(n_rows, n_cols, slim_versions, df_log, plot_title, var='rmse'):
    """
    Create and display a grid of evolution plots for multiple SLIM variations.
    Each subplot shows the mean and ±1 standard deviation envelope of a given metric (`var`)
    over generations (x-axis).

    Parameters
    ----------
    n_rows : int
        Number of rows in the subplot grid.
    n_cols : int
        Number of columns in the subplot grid.
    slim_versions : list
        List of SLIM versions to be plotted. Each identifier is used to filter the corresponding data from `df_log`.
    df_log : pandas.DataFrame
        DataFrame containing the log data for all runs. It must follow a specific structure:
        - Column 0: SLiM version
        - Column 4: Generation
        - Column 5: Train RMSE
        - Column 8: Validation RMSE
        - Column 9: Program size
    plot_title : str
        Title to display above the full grid of plots.
    var : str, optional
        The variable to plot.
        Either `'rmse'` (default) to show training and validation RMSE,
        or `'size'` to show the evolution of program size.

    Returns
    -------
    None
        Displays an interactive Plotly figure with the plots laid out in a grid.

    Notes
    -----
    - The envelope (±1 std) is visualized as a shaded area.
    - When `var='rmse'`, both training and validation curves are shown.
    """

    fig = make_subplots(
        rows=n_rows, cols=n_cols, 
        subplot_titles=[f'{i}' for i in slim_versions],
        vertical_spacing=0.1, horizontal_spacing=0.1
    )
    
    for i, sv in enumerate(slim_versions):
        row = i // n_cols + 1
        col = i % n_cols + 1
        show_legend = i == 0
        
        # Plot data
        df_plot = pd.DataFrame({
            'x': df_log[df_log[0]==sv].iloc[:, 4],
            'rmse': df_log[df_log[0]==sv].iloc[:, 5],
            'rmse_val': df_log[df_log[0]==sv].iloc[:, 8],
            'size': df_log[df_log[0]==sv].iloc[:, 9]
        })
        agg = df_plot.groupby('x')[var].agg(['mean', 'std']).reset_index()
        agg['y_upper'] = agg['mean'] + agg['std']
        agg['y_lower'] = agg['mean'] - agg['std']
        agg.loc[agg['y_lower'] < 0, 'y_lower'] = 0
    
        fig.add_trace(go.Scatter(
            x=agg['x'],
            y=agg['mean'],
            mode='lines',
            name='Train' if var=='rmse' else 'Size',
            line=dict(color='blue'),
            showlegend=show_legend
        ), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=agg['x'],
            y=agg['y_upper'],
            mode='lines',
            name='+1 std Train',
            line=dict(width=0),
            showlegend=False
        ), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=agg['x'],
            y=agg['y_lower'],
            mode='lines',
            name='-1 std Train',
            fill='tonexty',
            fillcolor='rgba(0,0,255,0.1)',
            line=dict(width=0),
            showlegend=False
        ), row=row, col=col)

        if var=='rmse':
            agg = df_plot.groupby('x')['rmse_val'].agg(['mean', 'std']).reset_index()
            agg['y_upper'] = agg['mean'] + agg['std']
            agg['y_lower'] = agg['mean'] - agg['std']
            fig.add_trace(go.Scatter(
                x=agg['x'],
                y=agg['mean'],
                mode='lines',
                name='Validation',
                line=dict(color='orange'),
                showlegend=show_legend
            ), row=row, col=col)
            fig.add_trace(go.Scatter(
                x=agg['x'],
                y=agg['y_upper'],
                mode='lines',
                name='+1 std Val',
                line=dict(width=0),
                showlegend=False
            ), row=row, col=col)
            fig.add_trace(go.Scatter(
                x=agg['x'],
                y=agg['y_lower'],
                mode='lines',
                name='-1 std Val',
                fill='tonexty',
                fillcolor='rgba(255,165,0,0.1)',
                line=dict(width=0),
                showlegend=False
            ), row=row, col=col)
        
    fig.update_layout(
        title_text=plot_title,
        xaxis_title='',
        yaxis_title='',
        height=700,
        width=1100,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.15,
            xanchor='center',
            x=0.5
        )
    )
    fig.update_yaxes(range=[0, None])
    fig.show()


def train_test_fit_and_size2(k_outer, k_inner, n_rows = 2, n_cols=3, model_name = 'SLIM-GSGP', size: bool = True):
    """
    Still not checked
    """

    df_log = []
    DATASET_NAME = model_name +'_sustavianfeed'
    LOG_DIR = './log/' + model_name + '/'
    for i_outer in range(k_outer):
        for i_inner in range(k_inner):
            LOG_PATH = LOG_DIR+DATASET_NAME+'_'+'outer'+'_'+str(i_outer)+'_'+'inner'+'_'+str(i_inner)+'.csv'
            tmp = pd.read_csv(LOG_PATH, header=None)
            comb_list = list(tmp[0].unique())
            assert n_rows*n_cols == len(tmp[0].unique()), 'n_rows times n_cols should be equal to the number of combinations tested on the grid search'
            tmp['cv'] = i_inner
            df_log.append(tmp)
    df_log = pd.concat(df_log, ignore_index=True)

    make_evolution_plots(n_rows, n_cols, comb_list, df_log,
                        plot_title = f'{model_name} - Train vs Test Fitness')
    if size:
        make_evolution_plots(n_rows, n_cols, comb_list, df_log, var='size',
                            plot_title = f'{model_name} - Size')


def test_best_combs(k_outer, model_name = 'SLIM-GSGP'):
    DATASET_NAME = model_name +'_sustavianfeed'
    LOG_DIR = './log/' + model_name + '/'
    rmse_by_config = {}
    for i in range(k_outer):
        test_all_rmse = []
        LOG_PATH = LOG_DIR+DATASET_NAME+'_'+'outer'+'_'+str(i)+'.csv'
        outer = pd.read_csv(LOG_PATH, header=None)
        config = outer[0].unique()[0]
        test_rmse = outer.iloc[-1, 8]
        test_all_rmse.append(test_rmse)
        if config not in rmse_by_config.keys():
            rmse_by_config[config] = test_all_rmse
        else:
            rmse_by_config[config].append(test_rmse)
    
    #print(rmse_by_config)

    fig = go.Figure()
    for config, rmse_values in rmse_by_config.items():
        fig.add_trace(go.Box(
            y=rmse_values,
            boxpoints='all',
            jitter=0.5,
            pointpos=0,
            line=dict(color='orange'),
            name=config
        ))

    fig.update_layout(
        title= model_name +' dataset',
        xaxis_title='',
        yaxis_title='Test RMSE',
        height=500, width=1100,
        yaxis_range=[0,None],
        margin=dict(l=50, r=50, t=50, b=20),
        showlegend=False,
        template='plotly_white'
    )

    fig.show()

def train_test_best_combs(k_outer, n_rows, n_cols, model_name = 'SLIM-GSGP'):
    DATASET_NAME = model_name +'_sustavianfeed'
    LOG_DIR = './log/' + model_name + '/'
    test_rmse_by_config = {}
    train_rmse_by_config = {}
    for i in range(k_outer):
        LOG_PATH = LOG_DIR+DATASET_NAME+'_'+'outer'+'_'+str(i)+'.csv'
        outer = pd.read_csv(LOG_PATH, header=None)
        config = outer[0].unique()[0]
        test_rmse = outer.iloc[-1, 8]
        train_rmse = outer.iloc[-1, 5]
        if config not in test_rmse_by_config.keys():
            test_rmse_by_config[config] = []
            train_rmse_by_config[config] = []
        
        test_rmse_by_config[config].append(test_rmse)
        train_rmse_by_config[config].append(train_rmse)
    
    assert n_rows*n_cols == len(test_rmse_by_config.keys())
    fig = make_subplots(
        rows=n_rows, cols=n_cols, 
        subplot_titles=[f'{i}' for i in test_rmse_by_config.keys()],
        vertical_spacing=0.1, horizontal_spacing=0.1
    )
    for i, config in enumerate(test_rmse_by_config.keys()):
        row = i // n_cols + 1
        col = i % n_cols + 1

        fig.add_trace(
        go.Scatter(y=train_rmse_by_config[config], mode='lines', name='Train', line=dict(color='orange'), text=config),
                row=row, col=col
                )
    
        fig.add_trace(
        go.Scatter(y=test_rmse_by_config[config], mode='lines', name='Test', line=dict(color='blue'), text=config)
                )

    fig.update_layout(
        title= model_name +' dataset',
        xaxis_title='',
        yaxis_title='RMSE',
        height=500, width=1100,
        yaxis_range=[0,None],
        margin=dict(l=50, r=50, t=50, b=20),
        showlegend=False,
        template='plotly_white'
    )

    fig.show()


"""
Vizualization functions not adapted to the implemtation of cv
"""

def train_test_fit(df, train_color='blue', test_color='orange', rows=5, cols=4):
    dif_combs = df[1].unique()  # Get unique combinations
    unique_setting_df = pd.DataFrame(dif_combs)
    num_plots = len(dif_combs)
    assert rows*cols==num_plots
    
    # Create subplot grid
    fig = sp.make_subplots(rows=rows, cols=cols, 
                           subplot_titles=[f"Combination index: {unique_setting_df[unique_setting_df[0]==comb].index[0]}" 
                                           for comb in dif_combs])
    
    for i, comb in enumerate(dif_combs):
        y = df[df[1] == comb]
        algo = y.iloc[0,0]
        row = (i // cols) + 1  #Calculate row position
        col = (i % cols) + 1   #Calculate column position
        
        fig.add_trace(
            go.Scatter(y=y.iloc[:, 5].values, mode='lines', name='Train', line=dict(color=train_color),
                       showlegend=(i==0)),
            row=row, col=col
        )
        
        fig.add_trace(
            go.Scatter(y=y.iloc[:, 8].values, mode='lines', name='Test', line=dict(color=test_color),
                       showlegend=(i==0)),
            row=row, col=col
        )

        fig.update_yaxes(range=[0, None], row=row, col=col)
    
    fig.update_layout(
        height=150 * rows,
        width=250 * cols,
        margin=dict(t=50),
        title_text=f'{algo} - Train vs Test Fitness (x=Generation, y=RMSE)',
        showlegend=True
    )

    fig.update_annotations(font_size=10)
    fig.show()


def train_test_fit_and_size1(df, comb_idxs: list | int = [i for i in range(pd.DataFrame(df[1].unique()).shape[0])],
                            train_color='blue', test_color='orange'):
     unique_setting_df = pd.DataFrame(df[1].unique())
     for comb_idx in comb_idxs:
          comb = unique_setting_df.iloc[comb_idx, 0]
          y = df[df[1]==comb]
          algo = y.iloc[0,0]
          fig = make_subplots(
          rows=1, cols=2,
          subplot_titles=(f'{algo} - Fitness evolution\nCombination:', f'{algo} - Size evolution')
          )

          fig.add_trace(go.Scatter(y=y.iloc[:,5].values, 
                                   mode='lines', name='Train', line=dict(color=train_color)), row=1, col=1)
          fig.add_trace(go.Scatter(y=y.iloc[:,8].values, 
                                   mode='lines', name='Test', line=dict(color=test_color)), row=1, col=1)
          fig.add_trace(go.Scatter(y=y.iloc[:,9].values, 
                                   mode='lines', name='Size'), row=1, col=2)
          
          fig.update_xaxes(title_text="Generation")

          fig.update_layout(
          width=1000,
          height=400, 
          showlegend=True,
          yaxis_range=[0,None],
          )
          fig.show()


def niche_entropy(df, train_color='blue', rows=5, cols=4):
    dif_combs = df[1].unique()  # Get unique combinations
    unique_setting_df = pd.DataFrame(dif_combs) # array to df
    num_plots = len(dif_combs)
    assert rows*cols==num_plots, "The number of combinations does not correspond to the grid size defined (rows/cols)."

    fig = sp.make_subplots(rows=rows, cols=cols, 
                           subplot_titles=[f"Combination index: {unique_setting_df[unique_setting_df[0]==comb].index[0]}" 
                                           for comb in dif_combs])

    for i, comb in enumerate(dif_combs):
        y = df[df[1] == comb]
        algo = y.iloc[0,0]
        row = (i // cols) + 1
        col = (i % cols) + 1
        
        fig.add_trace(
            go.Scatter(
                y=y.iloc[:, 10].values,
                mode='lines',
                name='Niche Entropy',
                line=dict(color=train_color),
                showlegend=(i == 0)), row=row, col=col
                )
    
    fig.update_layout(
        height=150 * rows,
        width=250 * cols,
        margin=dict(t=50),
        title_text=f'{algo} - Niche Entropy (x=Generation, y=Entropy)',
    )
    
    fig.show()


def plot_combs_together_test(df, comb_idxs: list | int = [i for i in range(pd.DataFrame(df[1].unique()).shape[0])],
                             colors = ['#FF0000', '#0000FF', '#00FF00', '#FFA500', '#800080', 
                                       '#FF00FF', '#00FFFF', '#FFFF00', '#1F77B4', '#FF7F0E',
                                       '#2CA02C', '#D62728', '#9467BD', '#8C564B', '#E377C2',
                                       '#7F7F7F', '#AEC7E8', '#FFBB78', '#98DF8A', '#FF9896'],
                              ):
     
     assert len(colors)>=len(comb_idxs), "Not enough colors for all combinations"

     unique_setting_df = pd.DataFrame(df[1].unique())
     fig = go.Figure()
     for i, comb_idx in enumerate(comb_idxs):
          comb = unique_setting_df.iloc[comb_idx, 0]
          y = df[df[1]==comb]
          algo = y.iloc[0,0]

          fig.add_trace(go.Scatter(y=y.iloc[:,8].values, 
                                   mode='lines', name=f'Test Comb {comb_idx}',
                                   line=dict(color=colors[i])))#, row=1, col=1)
          
          fig.update_xaxes(title_text="Generation")

     fig.update_layout(
          width=1000,
          height=400, 
          title_text = f"{algo} - Test Fitness (Combinations indexes: {comb_idxs})",
          showlegend=True,
          yaxis_range=[0,None],
          )
     
     fig.show()

