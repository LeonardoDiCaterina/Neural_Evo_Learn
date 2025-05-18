import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots

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
            title_text=f"{model_name} - Outer Fold {i_outer}",
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
