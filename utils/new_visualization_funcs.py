import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots

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
# 13 - The parameter combination string

def get_comb_values(model_name):
    LOG_PATH = './log/' + model_name + '/' + model_name +'_sustavianfeed' + '_outer_results.csv'
    df_results = pd.read_csv(LOG_PATH) 
    comb_train_values = {}
    comb_test_values = {}
    for comb in df_results['dynamic_params'].unique():
        df_one_comb = df_results[df_results['dynamic_params']==comb]
        train_values = df_one_comb['rmse_train']
        test_values = df_one_comb['rmse_test']
        comb_train_values[comb]=list(train_values)
        comb_test_values[comb]=list(test_values)

    return comb_train_values, comb_test_values


def test_best_combs(model_name):
    test_rmse_by_config = get_comb_values(model_name=model_name)[1]
    fig = go.Figure()
    for config, rmse_values in test_rmse_by_config.items():
        config_paragraphed = ',<br>'.join([part.strip() for part in config.split(',')])
        fig.add_trace(go.Box(
            y=rmse_values,
            boxpoints='all',
            jitter=0.5,
            pointpos=0,
            line=dict(color='orange'),
            name=config_paragraphed
        ))

    fig.update_layout(
        title= model_name +' - Best Combinations',
        xaxis_title='',
        yaxis_title='Test RMSE',
        height=500, width=1100,
        yaxis_range=[0,None],
        margin=dict(l=50, r=50, t=50, b=20),
        showlegend=False,
        template='plotly_white'
    )

    fig.show()


def train_test_best_combs(model_name):
    train_rmse_by_config = get_comb_values(model_name=model_name)[0]
    test_rmse_by_config = get_comb_values(model_name=model_name)[1]

    fig = make_subplots(
        rows=len(test_rmse_by_config.keys()), cols=1, 
        subplot_titles=[f'Combination:{i}' for i in test_rmse_by_config.keys()],
    )
    for i, config in enumerate(test_rmse_by_config.keys()):
        config_paragraphed = ',<br>'.join([part.strip() for part in config.split(',')])
        fig.add_trace(go.Box(
            y=train_rmse_by_config[config],
            boxpoints='all',
            jitter=0.5,
            pointpos=0,
            line=dict(color='orange'),
            name='Train'
        ), row=i+1, col=1)

        fig.add_trace(go.Box(
            y=test_rmse_by_config[config],
            boxpoints='all',
            jitter=0.5,
            pointpos=0,
            line=dict(color='blue'),
            name='Test'
        ), row=i+1, col=1)

    fig.update_layout(
        title_text=model_name + ' - Best Combinations',
        height=300 * len(test_rmse_by_config.keys()),  # Dynamic height based on number of subplots
        width=1100,
        margin=dict(l=50, r=50, t=100, b=50),  # Adjust margins
        template='plotly_white',
        showlegend=False  # Show legend if needed
    )
    
    # Update y-axis properties for all subplots
    fig.update_yaxes(
        title_text="RMSE",
        range=[0, None],
        showgrid=True
    )
    
    # Optionally adjust title positions
    fig.update_annotations(
        yshift=20  # Move subplot titles up slightly
    )

    fig.show()


def fit_and_size_per_outer(k_outer, model_name):
    LOG_DIR = './log/' + model_name + '/' + model_name +'_sustavianfeed'
    for i_outer in range(k_outer):
        LOG_PATH = LOG_DIR + f'_outer_{i_outer}.csv'
        df = pd.read_csv(LOG_PATH, header=None)
        param_str = df[13][0]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f'Fitness evolution', f'Size evolution'),
            horizontal_spacing =0.15
        )

        fig.add_trace(go.Scatter(y=df.iloc[:,5].values, 
                                mode='lines', name='Train', line=dict(color='orange')), row=1, col=1)
        fig.add_trace(go.Scatter(y=df.iloc[:,8].values, 
                                mode='lines', name='Test', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(y=df.iloc[:,9].values, 
                                mode='lines', name='Size'), row=1, col=2)
        fig.update_layout(
            width=1000,
            height=400, 
            showlegend=True,
            yaxis_range=[0,None],
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=-0.3,
                xanchor='center',
                x=0.5
            ),
            title_text=f"{model_name} - Outer Fold {i_outer} (Combination: {param_str})",
            title_x=0.5,  # Center title
            title_font=dict(size=18),  # Bigger font
            margin=dict(t=100)
        )

        fig.update_yaxes(
            title_text="RMSE",  # First y-axis title
            row=1, col=1
        )
        
        fig.update_yaxes(
            title_text="Size",
            type = "log",  #Logarithmic scale to handle the large numbers better
            tickformat=".1e",  # Scientific notation with 2 decimal places
            exponentformat="power",  # Shows as ×10ⁿ instead of en
            row=1, col=2
        )

        fig.show()



def make_evolution_plots(n_rows, n_cols, slim_versions, df_log, plot_title, var='rmse'):
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols, 
        subplot_titles=[f'{i}' for i in slim_versions],
        vertical_spacing=0.2, horizontal_spacing=0.2
    )
    
    for i, sv in enumerate(slim_versions):
        row = i // n_cols + 1
        col = i % n_cols + 1
        show_legend = i == 0
        
        # Plot data
        df_plot = pd.DataFrame({
            'x': df_log[df_log[13]==sv].iloc[:, 4],
            'rmse': df_log[df_log[13]==sv].iloc[:, 5],
            'rmse_val': df_log[df_log[13]==sv].iloc[:, 8],
            'size': df_log[df_log[13]==sv].iloc[:, 9]
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
        xaxis_title='Generations',
        yaxis_title='RMSE',
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


def fit_and_size_per_comb(k_outer, model_name, size=False):
    LOG_DIR = './log/' + model_name + '/' + model_name +'_sustavianfeed'
    df_log = [] # group all outers here
    comb_list = [] # log all unique combinations
    for i_outer in range(k_outer):
        df = pd.read_csv(LOG_DIR + f'_outer_{i_outer}.csv', header=None)
        df['cv'] = i_outer
        param_str = df[13][0]
        comb_list.append(param_str)
        df_log.append(df)
    df_log = pd.concat(df_log, ignore_index=True)

    unique_comb_list = list(set(comb_list))

    make_evolution_plots(n_rows=3, n_cols=1, slim_versions=unique_comb_list, df_log=df_log, plot_title = f'{model_name} - Train vs Test Fitness')
    
    if size:
        make_evolution_plots(3, 1, unique_comb_list, df_log, var='size', plot_title = 'SLIM - Size ('+model_name+' dataset)')


"""

For the next delivery


"""

def niche_entropy(k_outer, model_name, rows=5, cols=2):
    LOG_DIR = './log/' + model_name + '/' + model_name +'_sustavianfeed'

    assert rows*cols==k_outer, "The number of combinations does not correspond to the grid size defined (rows/cols)."

    fig = sp.make_subplots(rows=rows, cols=cols, 
                           #subplot_titles=[f"Combination index: {unique_setting_df[unique_setting_df[0]==comb].index[0]}" 
                           #                for comb in dif_combs]
                            )

    for i_outer in range(k_outer):
        LOG_PATH = LOG_DIR + f'_outer_{i_outer}.csv'
        df = pd.read_csv(LOG_PATH, header=None)
        y = df
        row = (i_outer // cols) + 1
        col = (i_outer % cols) + 1
        
        fig.add_trace(
            go.Scatter(
                y=y.iloc[:, 10].values,
                mode='lines',
                name='Niche Entropy',
                line=dict(color='orange'),
                showlegend=(i_outer == 0)), row=row, col=col
                )
    
        fig.update_layout(
            height=150 * rows,
            width=250 * cols,
            margin=dict(t=50),
            title_text=f'{model_name} - Niche Entropy (x=Generation, y=Entropy)',
        )
    
    fig.show()


def pop_fitness_diversity(k_outer, model_name):
    LOG_DIR = './log/' + model_name + '/' + model_name +'_sustavianfeed'
    for i_outer in range(k_outer):
        LOG_PATH = LOG_DIR + f'_outer_{i_outer}.csv'
        df = pd.read_csv(LOG_PATH, header=None)
        param_str = df[13][0]

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=df.iloc[:,11].values, 
                                mode='lines', name='Train', line=dict(color='orange')))
        fig.update_layout(
            height=400, width=800, 
            margin=dict(t=50),
            yaxis_range=[0,None],
            title_text=f'{model_name} - Population Fitness Diversity (Outer fold {i_outer}: Comb {param_str})',
            xaxis_title='Generation', yaxis_title='Fitness Standard Deviation'
        )
        fig.show()