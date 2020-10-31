import os

import numpy as np
import pandas as pd
import plac
from catboost import CatBoostRegressor, Pool
from catboost.utils import eval_metric
from sklearn.model_selection import train_test_split
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..',
))


def train(features, iterations=100, train_size=0.7):
    corpus_path = os.path.join(ROOT, 'corpus/calculated_metrics.csv')
    _df = pd.read_csv(corpus_path, index_col=0)

    mistakes = _df[(_df['review'] == 5) & (_df['true_mask_pixels'] == 0) & (_df['pred_mask_pixels'] != 0)]

    results = []
    results_rint = []
    models_data = []

    for _ in tqdm(range(iterations)):
        target = 'review'
        df = _df[features + [target]]

        df_train, df_test = train_test_split(df, train_size=train_size)

        train_pool = Pool(df_train[features], label=df_train[target])
        test_pool = Pool(df_test[features], label=df_test[target])

        cb_mae = CatBoostRegressor(loss_function='MAE', silent=True)
        cb_mae.fit(train_pool, eval_set=test_pool)

        models_data.append((cb_mae, df_test))

        cb_mae_pred = cb_mae.predict(test_pool)
        cb_mae_pred_rint = np.rint(cb_mae_pred)

        results.append(eval_metric(df_test[target].to_numpy(), cb_mae_pred, 'MAE'))
        results_rint.append(eval_metric(df_test[target].to_numpy(), cb_mae_pred_rint, 'MAE'))

    best_model_data = models_data[int(np.amin(results_rint))]

    return best_model_data


@plac.annotations(
    model_name=("Model name with extension. Default: cb_mae.cbm", "positional", None, str),
    iterations=("Number of iterations during training", "option", "i", int),
    train_size=("Train size during trainig", "option", "t", float),
)
def main(model_name=None, iterations=100, train_size=0.7):
    features = ['true_mask_pixels', 'pred_mask_pixels', 'DICE', 'TP', 'FP', 'REFVOL', 'MUTINF']
    model, test_data = train(features, iterations, train_size)

    pred = np.rint(model.predict(test_data))
    metric = eval_metric(test_data['review'].to_numpy(), pred, 'MAE')[0]

    model_name = model_name if model_name else f"cb_mae-{round(metric, 4)}.cbm"

    savepath = os.path.join(ROOT, 'models', 'catboost', model_name)
    model.save_model(savepath, format="cbm")

    print('Model was saved to ' + savepath)


if __name__ == "__main__":
    plac.call(main)
