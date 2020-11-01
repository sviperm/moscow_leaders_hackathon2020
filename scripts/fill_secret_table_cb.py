import os
import sys

import pandas as pd
import plac

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',))
sys.path.insert(0, ROOT)

from scripts.catboost_predict import get_models_by_names, predict  # nopep8


def fill_row(models):
    def fun(row):
        img_code = row[0][:-4]
        expert_path = os.path.join(ROOT, 'dataset', 'Expert', f'{img_code}_expert.png')
        sample1_path = os.path.join(ROOT, 'dataset', 'sample_1', f'{img_code}_s1.png')
        sample2_path = os.path.join(ROOT, 'dataset', 'sample_2', f'{img_code}_s2.png')
        sample3_path = os.path.join(ROOT, 'dataset', 'sample_3', f'{img_code}_s3.png')

        for i, sample_path in enumerate([sample1_path, sample2_path, sample3_path]):
            pred = predict(models, expert_path, sample_path)
            row[i + 1] = pred
        return row
    return fun


@plac.annotations(
    models_names=("Model name with extension. Default: cb_mae.cbm", "positional", None, str)
)
def main(models_names):
    models = get_models_by_names(models_names)

    table_path = os.path.join(ROOT, 'dataset', 'SecretPart_dummy.csv')
    table = pd.read_csv(table_path)
    table = table.apply(fill_row(models), axis=1)

    save_path = os.path.join(ROOT, 'submissions', 'SecretPart_Кибер-медики_cb.csv')
    table.to_csv(save_path, index=False)


if __name__ == "__main__":
    plac.call(main)
    # main('rf_2m-0.557.cbm, rf_2m-0.531.cbm, rf_2m-0.556.cbm, rf_2m-0.529.cbm, rf_2m-0.532.cbm')
