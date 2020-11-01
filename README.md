# Кибер-медики

## Установка
### Локально
```
$ apt-get update && apt-get install -y libinsighttoolkit4-dev
$ python -m venv venv
$ source venv/bin/activate
$ pip install -U pip
$ pip install -r requirements.txt
```
### Docker
```
$ docker build -t hackathon2020 .

$ docker run -p 8080:8080 %CONTAINRE_ID% -d
```
## Скрипты

### Скрипт предсказания Catboost 
Принимает на вход 3 аргумента:

1. Путь до маски эксперта
2. Путь до маски алгоритма
3. Название моделей, которые находятся в папке `models/catboost`. Обязательно в двойных ковычках, если несколько моделей, то через запятую. В случае если несколько моделей, то результат определяется голосванием.

```
$ python scripts/catboost_predict.py dataset/Expert/00000181_061_expert.png dataset/sample_1/00000181_061_s1.png "cb_mae-0.5818.cbm"
```

### Скрипт генерации Catboost модели
Принимает на вход 3 аргумента

1. Название модели (необязательно)
2. Количество итераций - целое число
3. Доля обучающих данных в выборке - дробное число
```
$ python scripts/generate_catboost_model.py -i 10 -t 0.7
```

### Скрипт заполнения таблицы, используя модели Catboost
Принимает на вход 1 аргумент - список моделей в двойных ковычках через запятую.
```
$ python scripts/fill_secret_table_cb.py "cb_mae-0.5818.cbm, cb_mae-0.6182_.cbm, cb_mae-0.6182.cbm, cb_mae-0.6364.cbm, cb_mae-0.6389.cbm"
```
## Research
### Нами были предложены и реализованы следующие идеи:
1. Использовать метрики для оценки сегментации медицинских изображений из
   [Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool] (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4533825/).
2. Выяснить корреляцию между заключением врача (выраженное числом от 1 до 5) и
   значениями рассчитанных метрик (использовался коэффициент ранговой
   корреляции Кендалла, как наиболее устойчивый к случайным вбросам,
   не требующий стандартного распределения случайных величин и
   показывющий монотонную связь между двумя переменными).
3. На основании коэффециента корреляции для каждой метрки, было произведено
   ранжирование метрик и отобраны наиболее оптимальные их них.
4. Отобранные метрики были использованы для построения следующих моделей:
    * [Cat Boost Regressor](https://catboost.ai/docs/concepts/python-reference_catboostregressor.html)
    * [Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
    * [Random Forest Regressor](https://scikit-learn.org/stable/search.html?q=random+forest)
    * [K Neighbors Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html?highlight=kneighborsregressor#sklearn.neighbors.KNeighborsRegressor)
    * [SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html?highlight=svr#sklearn.svm.LinearSVR)
    * [Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso)
5. Использовать сверточную нейронную сеть для вычисления метрики сходства двух
   масок на основании мнения эксперта. На вход подавались маска, размеченная
   ИИ, маска, размеченная экспертом и оригинальное изображение, объединенные в
   один массив. В качестве оптимизатора использовался алгортим Adam, функция
   ошибки -- Categorical Crossentropy.
   [Применялась простая архитектура](https://keras.io/examples/vision/mnist_convnet/) т.к данных для вычисления большого количества параметров не хватило.
