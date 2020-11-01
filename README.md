# Кибер-медики

## Установка
### Локально
```
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
$ python scripts/catboost_predict.py.py dataset/Expert/00000181_061_expert.png dataset/sample_1/00000181_061_s1.png "cb_mae-0.5818.cbm"
```

### Скрипт генерации Catboost модели
Принимает на вход 3 аргумента

1. Название модели (необязательно)
2. Количество итераций - целое число
3. Доля обучающих данных в выборке - дробное число
```
$ python scripts/generate_model.py -i 10 -t 0.7
```

### Скрипт заполнения таблицы, используя модели Catboost
Принимает на вход 1 аргумент - список моделей в двойных ковычках через запятую.
```
$ python scripts/fill_secret_table.py "cb_mae-0.5818.cbm, cb_mae-0.6182_.cbm, cb_mae-0.6182.cbm, cb_mae-0.6364.cbm, cb_mae-0.6389.cbm"
```