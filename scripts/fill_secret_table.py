import os
import re
import subprocess

import numpy as np
import pandas as pd
import plac
from catboost import CatBoostRegressor
from PIL import Image

ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..',
))


def most_common(lst):
    return max(set(lst), key=lst.count)


def calc_pixels(path):
    img = Image.open(path).convert('L')
    np_img = np.array(img)
    np_img[np_img > 0] = 1
    return np.count_nonzero(np_img)


def calculate_metrics(true_path, pred_path):
    evaluator_path = os.path.join(ROOT, 'scripts', 'evaluate')
    features = ['DICE', 'TP', 'FP', 'REFVOL', 'MUTINF']
    cmd_metrics = ','.join(features)

    metrics_str = subprocess.run([evaluator_path,
                                  true_path,
                                  pred_path,
                                  '-use', cmd_metrics],
                                 cwd=os.path.realpath(os.path.join(os.getcwd(), '..', '..')),
                                 capture_output=True)

    metrics_str = metrics_str.stdout.decode("utf-8").strip()

    metrics = re.findall(r"([A-Z]+)\s+=\s([\.\d]+)\s+[\w\(\)\-,\s]+\s?$",
                         metrics_str, re.MULTILINE)
    if not metrics:
        metrics = list(map(lambda x: (x, 0), features))

    return metrics


def fill_row(models):
    def fun(row):
        img_code = row[0][:-4]
        expert_path = os.path.join(ROOT, 'dataset', 'Expert', f'{img_code}_expert.png')
        sample1_path = os.path.join(ROOT, 'dataset', 'sample_1', f'{img_code}_s1.png')
        sample2_path = os.path.join(ROOT, 'dataset', 'sample_2', f'{img_code}_s2.png')
        sample3_path = os.path.join(ROOT, 'dataset', 'sample_3', f'{img_code}_s3.png')

        true_mask_pixels = calc_pixels(expert_path)

        features = ['true_mask_pixels', 'pred_mask_pixels', 'DICE', 'TP', 'FP', 'REFVOL', 'MUTINF']
        for i, sample_path in enumerate([sample1_path, sample2_path, sample3_path]):
            metrics = dict(calculate_metrics(expert_path, sample_path))

            data = pd.DataFrame({
                'true_mask_pixels': [true_mask_pixels],
                'pred_mask_pixels': [calc_pixels(sample_path)],
                'DICE': [metrics['DICE']],
                'TP': [metrics['TP']],
                'FP': [metrics['FP']],
                'REFVOL': [metrics['REFVOL']],
                'MUTINF': [metrics['MUTINF']],
            })

            preds = [round(model.predict(data[features])[0])
                     for model in models]

            if list(set(preds)) == preds:
                pred = round(np.mean(preds))
            else:
                pred = most_common(preds)

            row[i + 1] = pred
        return row
    return fun


@plac.annotations(
    models_names=("Model name with extension. Default: cb_mae.cbm", "positional", None, str)
)
def main(models_names):
    models_names = re.split(r',\s*', models_names)

    models = []
    for models_name in models_names:
        model_path = os.path.join(ROOT, 'models', 'catboost', models_name)
        model = CatBoostRegressor().load_model(model_path)
        models.append(model)

    table_path = os.path.join(ROOT, 'dataset', 'SecretPart_dummy.csv')
    table = pd.read_csv(table_path)
    table = table.apply(fill_row(models), axis=1)

    save_path = os.path.join(ROOT, 'submissions', 'SecretPart_Кибер-медики.csv')
    table.to_csv(save_path, index=False)


if __name__ == "__main__":
    plac.call(main)
    # main('cb_mae-0.6182.cbm')
