from typing import Sequence, Dict, Union, Set
import torch
import numpy as np
from edustudio.evaltpl import BaseEvalTPL
from edustudio.utils.common import tensor2npy
import pandas as pd


def doa_report(df):
    knowledges = []
    knowledge_item = []
    knowledge_user = []
    knowledge_truth = []
    knowledge_theta = []
    for user, item, score, theta, knowledge in df[["user_id", "item_id", "score", "theta", "knowledge"]].values:
        if isinstance(theta, list):
            for i, (theta_i, knowledge_i) in enumerate(zip(theta, knowledge)):
                if knowledge_i == 1: 
                    knowledges.append(i) # 知识点ID
                    knowledge_item.append(item) # Item ID
                    knowledge_user.append(user) # User ID
                    knowledge_truth.append(score) # score
                    knowledge_theta.append(theta_i) # matser
        else:  # pragma: no cover
            for i, knowledge_i in enumerate(knowledge):
                if knowledge_i == 1:
                    knowledges.append(i)
                    knowledge_item.append(item)
                    knowledge_user.append(user)
                    knowledge_truth.append(score)
                    knowledge_theta.append(theta)

    knowledge_df = pd.DataFrame({
        "knowledge": knowledges,
        # "user_id": knowledge_user,
        "item_id": knowledge_item,
        "score": knowledge_truth,
        "theta": knowledge_theta
    })

    knowledge_ground_truth = []
    knowledge_prediction = []
    for _, group_df in knowledge_df.groupby("knowledge"):
        _knowledge_ground_truth = []
        _knowledge_prediction = []
        for _, item_group_df in group_df.groupby("item_id"):
            _knowledge_ground_truth.append(item_group_df["score"].values)
            _knowledge_prediction.append(item_group_df["theta"].values)
        knowledge_ground_truth.append(_knowledge_ground_truth)
        knowledge_prediction.append(_knowledge_prediction)

    return doa_eval(knowledge_ground_truth, knowledge_prediction)

def doa_eval(y_true, y_pred):
    """
    >>> import numpy as np
    >>> y_true = [
    ...     [np.array([1, 0, 1])],
    ...     [np.array([0, 1, 1])]
    ... ]
    >>> y_pred = [
    ...     [np.array([.5, .4, .6])],
    ...     [np.array([.2, .3, .5])]
    ... ]
    >>> doa_eval(y_true, y_pred)['doa']
    1.0
    >>> y_pred = [
    ...     [np.array([.4, .5, .6])],
    ...     [np.array([.3, .2, .5])]
    ... ]
    >>> doa_eval(y_true, y_pred)['doa']
    0.5
    """
    doa = []
    doa_support = 0
    z_support = 0
    for knowledge_label, knowledge_pred in zip(y_true, y_pred):
        _doa = 0
        _z = 0
        for label, pred in zip(knowledge_label, knowledge_pred): # 每个习题
            if sum(label) == len(label) or sum(label) == 0:
                continue
            pos_idx = []
            neg_idx = []
            for i, _label in enumerate(label): # 找出所有(1, 0) pair
                if _label == 1:
                    pos_idx.append(i)
                else:
                    neg_idx.append(i)
            pos_pred = pred[pos_idx]
            neg_pred = pred[neg_idx]
            invalid = 0
            for _pos_pred in pos_pred:
                _doa += len(neg_pred[neg_pred < _pos_pred])
                invalid += len(neg_pred[neg_pred == _pos_pred])
            _z += (len(pos_pred) * len(neg_pred)) - invalid
        if _z > 0:
            doa.append(_doa / _z)
            z_support += _z # 有效pair个数
            doa_support += 1 # 有效doa
    return {
        "doa": np.mean(doa),
        "doa_know_support": doa_support,
        "doa_z_support": z_support,
        "doa_list": doa,
    }

class StuEmbEvalTPL(BaseEvalTPL):
    def eval(self, stu_stats=None, **kwargs):
        if stu_stats is None: return {}
        
        user_emb_npy = stu_stats
        datafmt = self.train_loader.dataset
        df_user = pd.DataFrame.from_dict({uid:str(list(user_emb_npy[uid, :])) for uid in range(user_emb_npy.shape[0])}, orient='index', columns=['theta']).reset_index().rename(columns={'index': 'stu_id:token'})
        df_user['theta'] = df_user['theta'].apply(lambda x: eval(x))

        df_Q = datafmt.df_exer[['exer_id:token', 'cpt_seq:token_seq']]
        df_Q['knowledge'] = [datafmt.Q_mat[i,:].tolist() for i in df_Q['exer_id:token']]

        df = datafmt.df.merge(df_user, on=['stu_id:token']).merge(df_Q, on=['exer_id:token'])
        df = df.rename(columns={'stu_id:token': 'user_id', 'exer_id:token':'item_id', 'label:float': 'score'})
        official_doa = doa_report(df)['doa']

        return {"doa": official_doa}
