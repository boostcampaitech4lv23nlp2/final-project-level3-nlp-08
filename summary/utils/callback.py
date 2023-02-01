from transformers import TrainerCallback
from sklearn.manifold import TSNE
import wandb
import plotly.express as px 
import wandb
import pandas as pd
from .utils import sub_label_list, sub_num_to_labels

class SubjectCallback(TrainerCallback):
    def __init__(self):
        self.eval_step = 0
        self.tSNE = TSNE(n_components=2, perplexity = 40)
        self.html_list = []

    def on_evaluate(self, args, state, control, **kwargs):
        self.eval_step += 1
        embedding = state.log_history[-1]['eval_embedding']
        labels = state.log_history[-1]['eval_labels']
        answer = state.log_history[-1]['eval_answer'] 
        preds = state.log_history[-1]['eval_preds']  
        # 안하면 이후 직렬화 오류남.
        del(state.log_history[-1]['eval_embedding'])
        del(state.log_history[-1]['eval_labels'])
        del(state.log_history[-1]['eval_answer'])
        del(state.log_history[-1]['eval_preds'])
        
        data = [[label, val] for (label, val) in zip(sub_label_list, answer)]
        acc_table = wandb.Table(data=data, columns = ["label", "acc"])
        wandb.log({"acc_table": acc_table})

        tSNE_result = self.tSNE.fit_transform(embedding)
        df = pd.DataFrame()
        df['x'] = tSNE_result[:, 0]
        df['y'] = tSNE_result[:, 1]
        df['label'] = sub_num_to_labels(labels)
        df['pred'] = sub_num_to_labels(preds)
        fig =px.scatter(df,
                        x='x',
                        y='y',
                        color='label',
                        hover_data=['pred'],
                        size=None            
                )
        path_to_plotly_html = f"tSNE.html"
        fig.write_html(path_to_plotly_html, auto_play = False)
        self.html_list.append(wandb.Html(path_to_plotly_html))
        # self.html_list이 커질 경우 느려지는 문제가 있음.
        wandb.log({"t-SNE": wandb.Table(data = [[i] for i in self.html_list], 
                                        columns = ['tSNE'])})