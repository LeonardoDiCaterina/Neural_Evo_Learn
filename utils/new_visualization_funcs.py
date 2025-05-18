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
        title_text=model_name + ' dataset',
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
