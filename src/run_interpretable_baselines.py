import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__))+"/../")

from edustudio.quickstart import run_edustudio
from eval_tpl import StuEmbEvalTPL
from data_tpl import DCDDataTPL
from train_tpl import DCDTrainTPL


run_edustudio(
    dataset='Matmat',
    datatpl_cfg_dict={
        'cls': DCDDataTPL,
        'M2C_FillMissingQ': {'Q_fill_type': 'sim_dist_for_by_exer'}
    },
    modeltpl_cfg_dict={
        'cls': 'KaNCD',
    },
    traintpl_cfg_dict={
        'cls': DCDTrainTPL,
        'seed': 2023,
        'epoch_num': 400,
        'lr': 0.001,
        'num_workers': 0,
        'batch_size': 1024,
        'num_stop_rounds': 10,
        'early_stop_metrics': [('auc','max'), ('doa', 'max')],
        'best_epoch_metric': 'doa',
    },
    evaltpl_cfg_dict={
        'clses': ['BinaryClassificationEvalTPL',  StuEmbEvalTPL],
    }
)
