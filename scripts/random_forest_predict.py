import os
import re
import subprocess

import joblib
import numpy as np
import pandas as pd
import plac
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',))


def most_common(lst):
    return max(set(lst), key=lst.count)


def get_models_by_names(models_names):
    models_names = re.split(r',\s*', models_names)

    models = []
    for models_name in models_names:
        model_path = os.path.join(ROOT, 'models', 'random_forest', models_name)
        model = joblib.load(model_path)
        models.append(model)

    return models


def calculate_metrics(true_path, pred_path):
    evaluator_path = os.path.join(ROOT, 'scripts', 'evaluate')
    features = ['HDRFDST', 'AVGDIST']
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


def predict(models, expert_path, sample_path, metrics=None):
    features = ['true_mask_pixels', 'pred_mask_pixels', 'HDRFDST', 'AVGDIST']
    if not metrics:
        metrics = dict(calculate_metrics(expert_path, sample_path))

    data = pd.DataFrame({
        'true_mask_pixels': [calc_pixels(expert_path)],
        'pred_mask_pixels': [calc_pixels(sample_path)],
        'HDRFDST': [metrics['HDRFDST']],
        'AVGDIST': [metrics['AVGDIST']],
    })

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


if __name__ == "__main__":
    # plac.call(main)
    main("dataset/Expert/00000181_061_expert.png",
         "dataset/sample_1/00000181_061_s1.png",
         "rf_2m-0.557.cbm, rf_2m-0.531.cbm, rf_2m-0.556.cbm, rf_2m-0.529.cbm, rf_2m-0.532.cbm")
