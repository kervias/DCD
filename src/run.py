import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__))+"/../")

from edustudio.quickstart import run_edustudio
from eval_tpl import StuEmbEvalTPL
from data_tpl import DCDDataTPL
from model_tpl import DCD
from train_tpl import DCDTrainTPL


run_edustudio(
    dataset='Matmat',
    datatpl_cfg_dict={
        'cls': DCDDataTPL
    },
    modeltpl_cfg_dict={
        'cls': DCD,
        'EncoderUserHidden': [768],
        'EncoderItemHidden': [768],
        'lambda_main': 1.0,
        'lambda_q': 10.0,
        'b_sample_type': 'gumbel_softmax',
        'b_sample_kwargs': {'tau': 1.0, 'hard': True},
        'align_margin_loss_kwargs': {'margin': 0.7, 'topk': 2, 'd1': 1, 'margin_lambda': 0.5, 'norm': 2, 'norm_lambda': 0.5, 'start_epoch': 1},
        'beta_user': 0.0,
        'beta_item': 0.0,
        'g_beta_user': 1.0,
        'g_beta_item': 1.0,
        'alpha_user': 0.0,
        'alpha_item': 0.0,
        'gamma_user': 1.0,
        'gamma_item': 1.0,
        'sampling_type': 'mws',
        'bernoulli_prior_p': 0.2,
    },
    traintpl_cfg_dict={
        'cls': DCDTrainTPL,
        'seed': 2023,
        'epoch_num': 400,
        'lr': 0.0005,
        'num_workers': 0,
        'batch_size': 2048,
        'num_stop_rounds': 10,
        'early_stop_metrics': [('auc','max'), ('doa', 'max')],
        'best_epoch_metric': 'doa',
    },
    evaltpl_cfg_dict={
        'clses': ['BinaryClassificationEvalTPL', StuEmbEvalTPL],
    }
)
