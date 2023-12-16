from typing import List

import pandas as pd
import numpy as np
from sklearn.inspection import partial_dependence
from sklearn.linear_model import LinearRegression
from plotly import graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_white"
from rich.progress import track

class ModelFingerprint:
    
    def __init__(self):
        self.feature_names = None
        self.lin_nonlin_effect = None
        self.pairwise_effect = None
        self.explained = False
        
    def plot(self):
        if not self.explained:
            raise Exception('Please run explain() first.')
        linear_nonlinear_df = self.lin_nonlin_effect
        pairwise_df = self.pairwise_effect
        
        # Plot linear and nonlinear effects
        trace1 = go.Bar(x=linear_nonlinear_df.index, y=linear_nonlinear_df['linear_effect'], name='Linear Effect')
        trace2 = go.Bar(x=linear_nonlinear_df.index, y=linear_nonlinear_df['nonlinear_effect'], name='Nonlinear Effect')
        fig = go.Figure([trace1, trace2])
        fig.update_layout(title='Linear and Nonlinear Effects', xaxis_title='Feature', yaxis_title='Effect')
        fig.show()
        
        # Plot pairwise effects
        x_axis = [f"{x}, {y}" for x, y in zip(pairwise_df['feat_x'], pairwise_df['feat_y'])]
        trace1 = go.Bar(x=x_axis, y=pairwise_df['pairwise_effect'], name='Pairwise Effect')
        fig = go.Figure([trace1])
        fig.update_layout(title='Pairwise Effects', xaxis_title='Feature Pair', yaxis_title='Effect')
        fig.show()
    
    def explain(self, model, explained_data: np.array, feature_names: List[str], grid_resolution: int=50, pairwise_combinations: None|List[tuple]=None):
        """

        Args:
            model: any model with sklearn-like API. PyTorch models could be wrapped with skorch.
            explained_data: data used to explain the model
            feature_names: list of feature names
            grid_resolution: the number of points that are used to generate the partial dependence plot for each feature
            pairwise_combinations: list of tuples of feature names for which the pairwise effect is computed
        """
        self.feature_names = feature_names

        pairwise_combinations = pairwise_combinations or []
        n_explained = 10_000 if explained_data.shape[0] > 10_000 else explained_data.shape[0]
        explained_data = explained_data[np.random.choice(explained_data.shape[0], n_explained, replace=False)]
        
        # Individual inear and nonlinear effects for each feature
        lin_nonlin_effect_df = pd.DataFrame(columns=['linear_effect', 'nonlinear_effect'])
        avg_pred_dict = {}
        for i, feat in enumerate(feature_names):
            res = partial_dependence(estimator=model, 
                                     X=explained_data,
                                     features=[i],  # feature for which the partial dependency is computed
                                     feature_names=feature_names, 
                                     grid_resolution=grid_resolution  # number of points on the grid
                                     )
            # decompose partial dependence result into linear and nonlinear effects
            grid_val = res.grid_values[0]  # value with which the grid was generated
            avg_pred = res.average[0]  # the averaged predictions
            lin_reg = LinearRegression()
            lin_reg.fit(grid_val.reshape(-1, 1), avg_pred)
            lin_pred = lin_reg.predict(grid_val.reshape(-1, 1))
            linear_effect = np.abs(lin_pred - avg_pred.mean()).mean()
            nonlinear_effect = np.abs(avg_pred - lin_pred).mean()
            
            avg_pred_dict[feat] = avg_pred  # later used for pairwise effects
            lin_nonlin_effect_df.loc[feat] = [linear_effect, nonlinear_effect]

        # Pairwise effects
        pairwise_combinations_idx = [(feature_names.index(x1), feature_names.index(x2)) for x1, x2 in pairwise_combinations]
        pairwise_effect_df = pd.DataFrame(columns=['feat_x', 'feat_y', 'pairwise_effect'])
        
        for feat1_idx, feat2_idx in track(pairwise_combinations_idx):
            res = partial_dependence(estimator=model, 
                                     X=explained_data,
                                     features=[feat1_idx, feat2_idx],  # feature for which the partial dependency is computed
                                     feature_names=feature_names, 
                                     grid_resolution=grid_resolution  # number of points on the grid
                                     )
            grid_val1 = res.grid_values[0]
            grid_val2 = res.grid_values[1]
            avg_pred = res.average[0]
            
            avg_pred1 = avg_pred_dict[feature_names[feat1_idx]]
            avg_pred2 = avg_pred_dict[feature_names[feat2_idx]]
            
            # demean
            avg_pred = avg_pred - avg_pred.mean()
            avg_pred1 -= avg_pred1.mean()
            avg_pred2 -= avg_pred2.mean()
            
            pairwise_effect_ls = []
            for i, feat1 in enumerate(grid_val1):
                for j, feat2 in enumerate(grid_val2):
                    pairwise_effect_ls.append(avg_pred[i, j] - avg_pred1[i] - avg_pred2[j])
            pairwise_effect = np.abs(pairwise_effect_ls).mean()
            new_val = {'feat_x': feature_names[feat1_idx], 
                    'feat_y': feature_names[feat2_idx], 
                    'pairwise_effect': pairwise_effect}
            pairwise_effect_df.loc[len(pairwise_effect_df)] = new_val
            
        self.lin_nonlin_effect = lin_nonlin_effect_df
        self.pairwise_effect = pairwise_effect_df
        self.explained = True