from edustudio.atom_op.mid2cache import M2C_RandomDataSplit4CD
from sklearn.model_selection import StratifiedKFold, KFold


class M2C_SplitDataset(M2C_RandomDataSplit4CD):
    def multi_fold_split(self, df):
        skf = StratifiedKFold(n_splits=int(self.n_folds), shuffle=True, random_state=self.m2c_cfg['seed'])
        splits = skf.split(df, df['stu_id:token'])

        train_list, test_list = [], []
        for train_index, test_index in splits:
            train_df = df.iloc[train_index].reset_index(drop=True)
            test_df = df.iloc[test_index].reset_index(drop=True)
            train_list.append(train_df)
            test_list.append(test_df)
        return train_list, test_list
