import os
import re
import subprocess

import numpy as np
import pandas as pd
import plac
from catboost import CatBoostRegressor
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',))


def most_common(lst):
    return max(set(lst), key=lst.count)


def get_models_by_names(models_names):
    models_names = re.split(r',\s*', models_names)

    models = []
    for models_name in models_names:
        model_path = os.path.join(ROOT, 'models', 'catboost', models_name)
        model = CatBoostRegressor().load_model(model_path)
        models.append(model)

    return models


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


def calc_pixels(path):
    img = Image.open(path).convert('L')
    np_img = np.array(img)
    np_img[np_img > 0] = 1
    return np.count_nonzero(np_img)


def predict(models, expert_path, sample_path):
    features = ['true_mask_pixels', 'pred_mask_pixels', 'DICE',
                'TP', 'FP', 'REFVOL', 'MUTINF']
    metrics = dict(calculate_metrics(expert_path, sample_path))

    data = pd.DataFrame({
        'true_mask_pixels': [calc_pixels(expert_path)],
        'pred_mask_pixels': [calc_pixels(sample_path)],
        'DICE': [metrics['DICE']],
        'TP': [metrics['TP']],
        'FP': [metrics['FP']],
        'REFVOL': [metrics['REFVOL']],
        'MUTINF': [metrics['MUTINF']],
    })

    # TODO убедиться в том, что все праивльно работает
    preds = [round(model.predict(data[features])[0])
             for model in models]

    if list(set(preds)) == preds:
        pred = round(np.mean(preds))
    else:
        pred = most_common(preds)

    return pred


@plac.annotations(
    expert_path=("M", "positional", None, str),
    sample_path=("M", "positional", None, str),
)
def main(expert_path, sample_path, models_names):
    models = get_models_by_names(models_names)
    pred = predict(models, expert_path, sample_path)
    print(f"Result: {pred}")
    return pred


if __name__ == "__main__":
    plac.call(main)
