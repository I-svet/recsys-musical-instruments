"""Создать исходный ноутбук. Выполнение: python scripts/execute_notebook.py."""
from pathlib import Path
import textwrap

import nbformat as nbf

root = Path(__file__).resolve().parents[1]
cells = []
def md(text):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))
def code(text):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(text).strip()))

md("""
# Рекомендации Amazon Musical Instruments 2018

Светлов Иван Ильич

Пять моделей с нуля: ItemKNN, ALS для явных оценок, EASE, FM с метапризнаками и timeSVD++.
Ещё сравниваются популярные товары, FM только с ID и SVD++ без времени.

## Запуск

Python 3.12, процессор, желательно 16 ГБ памяти. Из корня репозитория:

```bash
cd hw1
python -m pip install -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e .
python scripts/execute_notebook.py
```

Ноутбук можно выполнять и по ячейкам в Jupyter. Данные скачиваются из первоисточника.
Первый запуск выполняет весь поиск и обучение, это дольше просмотра готовых результатов.
Повторный запуск использует только завершённые совместимые этапы.
Выводы ячеек сохранены в этом файле. Данные и веса в Git не входят.
""")
code("""
from pathlib import Path
import os, sys, json, random, gc, subprocess, inspect
ROOT = Path.cwd().resolve()
if not (ROOT / 'models').is_dir():
    ROOT = ROOT / 'hw1'
os.chdir(ROOT)
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'artifacts/matplotlib'))
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Image
from threadpoolctl import threadpool_limits
from recsys.experiment import Experiment, configure_runtime
from recsys.data import prepare, load_dataset
from recsys.download import download_ratings, fetch, METADATA_URL, METADATA_SHA256, prepare_metadata
from recsys.io import sha256
from models.itemknn import ItemKNN
from models.als import ExplicitALS
from models.ease import EASE
from models.fm import FM
from models.timesvd import TimeSVDPP
%matplotlib inline
SEED = 2026
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
configure_runtime(4)
config = json.loads(Path('configs/benchmark.json').read_text(encoding='utf-8'))
PRIMARY_METRIC = 'ndcg@10'
K = 10
DATA = Path('data/prepared')
RUNS = Path('artifacts/hw1')
REPORT = Path('reports')
pd.set_option('display.max_colwidth', 180)
print('Основная метрика:', PRIMARY_METRIC, 'K:', K)
print('Начальные значения генераторов:', config['seeds'])
print('Python:', sys.version.split()[0], 'NumPy:', np.__version__, 'PyTorch:', torch.__version__)
""")
md("""
## 1. Данные и задача

[Amazon Review Data 2018, UCSD](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/), категория Musical Instruments.
Используются полный [файл оценок](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Musical_Instruments.csv)
и [метаданные товаров](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Musical_Instruments.json.gz).

| Поле | Смысл |
|---|---|
| item_id | ASIN товара |
| user_id | Идентификатор автора отзыва |
| rating | Оценка от 1 до 5 |
| timestamp | Время отзыва в секундах UTC |
| title, brand, categories | Название, бренд и категории из отдельного файла |

Один отзыв считается наблюдаемым взаимодействием. Время покупки неизвестно.
Рекомендуются 10 ещё не оценённых пользователем товаров.
Релевантный товар имеет будущую оценку не ниже 4. Для товаров без отзыва предпочтение неизвестно.
Этот критерий одинаков для всех моделей, хотя цели обучения у них разные.
""")
code("""
raw_path = download_ratings(Path('data/raw'))
meta_path = Path('data/raw/meta_Musical_Instruments.json.gz')
fetch(METADATA_URL, meta_path, METADATA_SHA256)
raw = pd.read_csv(raw_path, header=None, names=['item_id', 'user_id', 'rating', 'timestamp'],
                  dtype={'item_id': 'string', 'user_id': 'string'})
raw_shape = {'Взаимодействия': len(raw), 'Пользователи': raw.user_id.nunique(), 'Товары': raw.item_id.nunique()}
display(pd.DataFrame([raw_shape], index=['Исходный файл']))
print('SHA256 оценок:', sha256(raw_path))
print('Период:', pd.to_datetime(raw.timestamp.min(), unit='s', utc=True).date(),
      pd.to_datetime(raw.timestamp.max(), unit='s', utc=True).date())
""")
md("""
## 2. Анализ и подготовка

Сначала проверяются пропуски и допустимые оценки. Повторы пары пользователь-товар удаляются,
остаётся первый отзыв по времени. При совпадении дат берётся меньший номер исходной строки.
Все события одного дня остаются вместе, так как более точного времени нет.
""")
code("""
missing = raw.isna().sum().rename('Пропуски').to_frame()
display(missing)
invalid = raw.isna().any(axis=1) | ~raw.rating.isin([1, 2, 3, 4, 5]) | (raw.timestamp < 0)
print('Недопустимые строки:', int(invalid.sum()))
print('Полные дубликаты:', int(raw.duplicated().sum()))
print('Повторы пары пользователь-товар:', int(raw.duplicated(['user_id', 'item_id']).sum()))
print('Все даты имеют точность до дня:', bool((raw.timestamp % 86400 == 0).all()))
fig, axes = plt.subplots(1, 2, figsize=(11, 3.7))
raw.rating.value_counts().sort_index().plot.bar(ax=axes[0], rot=0, color='steelblue')
axes[0].set(xlabel='Оценка', ylabel='Число отзывов', title='Распределение оценок')
months = pd.to_datetime(raw.timestamp, unit='s', utc=True).dt.strftime('%Y-%m')
months.value_counts().sort_index().plot(ax=axes[1], color='steelblue')
axes[1].set(xlabel='Месяц UTC', ylabel='Число отзывов', title='Отзывы по времени')
plt.tight_layout()
plt.show()
""")
md("""
Пользователи и товары с совсем короткой историей дают мало информации соседским моделям.
Поэтому после временного разбиения только в обучающей части по очереди удаляются пользователи
и товары с менее чем пятью взаимодействиями. Отбор повторяется до устойчивого состава.
Полученные ID фиксируются для остальных частей. Будущая активность на этот отбор не влияет.
""")
code("""
manifest = prepare(raw_path, DATA)
data = load_dataset(DATA)
prepare_metadata(meta_path, data.items, DATA)
metadata = pd.read_csv(DATA / 'metadata.csv').fillna('')
parts = {'Обучение': data.train, 'Проверка': data.validation, 'Тест': data.test}
volume = pd.DataFrame([
    {'Часть': name, 'Взаимодействия': len(frame), 'Пользователи': frame.user.nunique(),
     'Товары': frame.item.nunique(), 'Положительные оценки': int((frame.rating >= 4).sum())}
    for name, frame in parts.items()
]).set_index('Часть')
display(volume)
print('Итого:', sum(map(len, parts.values())), 'взаимодействий,', len(data.users), 'пользователей,', len(data.items), 'товаров')
assert 100_000 <= sum(map(len, parts.values())) <= 2_000_000
assert len(data.users) >= 1000 and len(data.items) >= 1000
display(data.train.head())
del raw, months
gc.collect()
""")
code("""
user_activity = data.train.groupby('user').size()
item_popularity = data.train.groupby('item').size()
display(pd.DataFrame({'Отзывы пользователя': user_activity.describe(), 'Отзывы товара': item_popularity.describe()}))
filled = len(data.train) / (len(data.users) * len(data.items))
print(f'Заполнено {filled:.4%} обучающей матрицы. Разреженность {1-filled:.4%}')
fig, axes = plt.subplots(1, 2, figsize=(11, 3.7))
for ax, values, title in zip(axes, [user_activity, item_popularity], ['Активность пользователей', 'Популярность товаров']):
    ax.hist(values, bins=np.geomspace(5, max(values.max(), 6)+1, 30), color='steelblue')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set(xlabel='Число отзывов', ylabel='Число пользователей или товаров', title=title)
plt.tight_layout()
plt.show()
""")
md("""
Большая часть матрицы пустая, а активность неравномерная.
Данные остаются разреженными и после отбора. Поэтому пропуски нельзя просто считать плохими оценками.

Метаданные присоединяются слева. Отсутствие названия или бренда не удаляет взаимодействие.
Каталожный файл собран позднее части отзывов. Названия используются для показа рекомендаций,
а для FM берутся признаки, которые можно посчитать по прошлой истории.
""")
code("""
def categories(value):
    result = []
    def visit(x):
        if isinstance(x, list):
            for item in x:
                visit(item)
        elif x:
            result.append(str(x))
    visit(json.loads(value) if value else [])
    return result
meta_missing = pd.Series({col: int(metadata[col].eq('').sum()) for col in ['title', 'brand']}, name='Пустые поля')
meta_categories = metadata.categories.map(categories)
meta_missing.loc['categories'] = int(meta_categories.map(len).eq(0).sum())
display(meta_missing.to_frame())
display(metadata[['asin', 'title', 'brand', 'categories']].head())
leaf = meta_categories.map(lambda x: x[-1] if x else 'Нет категории')
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
metadata.brand.replace('', 'Нет бренда').value_counts().head(8).sort_values().plot.barh(ax=axes[0], color='steelblue')
leaf.value_counts().head(8).sort_values().plot.barh(ax=axes[1], color='steelblue')
axes[0].set_title('Частые бренды')
axes[1].set_title('Частые категории')
axes[0].set_ylabel('Бренд')
axes[1].set_ylabel('Категория')
for ax in axes:
    ax.set_xlabel('Число товаров')
plt.tight_layout()
plt.show()
""")
md("""
## 3. Протокол оценки

Границы выбраны до обучения по 80-му и 90-му процентилям времени полной категории после удаления повторов.

| Часть | Период UTC |
|---|---|
| Обучение | До 2017-04-17 |
| Проверка | С 2017-04-17 до 2017-12-08 |
| Тест | С 2017-12-08 до конца данных |

Для пользователя ранжируется весь фиксированный каталог из 7 329 товаров за вычетом всех уже оценённых,
включая оценки 1, 2 и 3. Рекомендация выдаётся на начало периода. События внутри периода не добавляются в историю.
Одинаковые оценки разрешаются по меньшему ID товара. Все модели используют одни и те же списки пользователей и кандидатов.

В основную оценку входят пользователи с положительными будущими оценками внутри каталога.
Новые пользователи и товары, а также ID вне отбора по обучающей части в основную таблицу не входят.
Это сравнение для активных пользователей и известного каталога, качество нового пользователя оно не измеряет.
Если у выбранного пользователя нет персонального сигнала, ему предлагаются популярные товары, и он остаётся в метриках.
Дополнительный Recall_all_future@10 включает товары вне каталога в знаменатель и пользователей с положительными отзывами только вне него.

До подбора основной метрикой выбран NDCG@10. Первые места списка важнее последних, поэтому учитывается позиция попадания.
Recall@10 показывает, какую долю положительных будущих товаров удалось найти в десяти рекомендациях.
Также считаются Precision@10 и MAP@10. Все метрики усредняются по пользователям с непустым множеством релевантных товаров.

Для бинарной релевантности: DCG@10 = Σ h_j / log₂(j+1), NDCG@10 = DCG@10 / IDCG@10.
Recall@10 = число попаданий / число релевантных товаров. Precision@10 = число попаданий / 10.
AP@10 = Σ h_j Precision@j / min(10, число релевантных товаров), MAP это среднее AP.
""")
code("""
train_snapshot = data.snapshot('validation')
test_snapshot = data.snapshot('test')
assert data.train.timestamp.max() < data.validation_start
assert data.validation.timestamp.min() >= data.validation_start
assert data.validation.timestamp.max() < data.test_start
assert data.test.timestamp.min() >= data.test_start
assert not pd.concat(parts.values()).duplicated(['user', 'item']).any()
display(pd.DataFrame([
    {'Часть': {'validation':'Проверка', 'test':'Тест'}[split], 'Основная группа': len(data.evaluation_users(split)),
     'С учётом товаров вне каталога': len(data.evaluation_users(split, include_outside=True))}
    for split in ['validation', 'test']
]))
display(pd.DataFrame(manifest['exclusions']).T)
print('Правила времени и уникальности пар проверены')
""")
md("""
## 4. Модели и подбор параметров

Код обучения и предсказания находится в отдельных файлах `models/`.

| Модель | Оценка и цель |
|---|---|
| ItemKNN | Косинусное сходство положительных историй со сглаживанием n_ij/(n_ij+gamma). Оценка X_u W, диагональ связей нулевая |
| ALS | r̂_ui = μ + b_u + b_i + p_uᵀq_i. Сумма квадратов ошибок по наблюдаемым оценкам и L2. Поочерёдно решаются задачи для пользователей и товаров |
| EASE | min ||X-XB||² + λ||B||² при diag(B)=0. P=(XᵀX+λI)⁻¹, B_ij=-P_ij/P_jj для i≠j |
| FM | w₀+Σw_jx_j+½Σ_f[(Σ_jv_jfx_j)²-Σ_jv_jf²x_j²]. Средняя ошибка квадратов и средний квадрат параметров |
| timeSVD++ | К смещениям и факторам добавлены интервалы времени товара, изменение предпочтений пользователя, дневные поправки и сумма векторов прошлой истории |

У timeSVD++ d_u(t)=sign(t-mean_time_u)|t-mean_time_u|^β, β=0,4.
h_u(t)=Σ y_j/√|H_u(<t)|. В историю входят только предыдущие дни.
Прогноз: μ+b_i+b_i,bin(t)+b_u+α_ud_u(t)+b_u,day(t)+q_iᵀ[p_u+a_ud_u(t)+p_u,day(t)+h_u(t)].
Неизвестные дневные поправки и интервалы товара равны нулю.

FM использует ID, логарифмы количества прошлых отзывов пользователя и товара, их средние оценки,
возраст истории и месяц с днём недели запроса. Эти признаки описывают активность и склонность ставить высокие оценки.
Данные текущего дня не входят в свои агрегаты. Для пустой истории число и возраст равны нулю, средняя оценка равна 3.
Масштабирование настраивается только по обучающим данным. Снимок названий, брендов и категорий в признаки не входит.

MostPopular нужен как простой ориентир. FM только с ID проверяет пользу метапризнаков.
SVD++ с отключённым временем проверяет вклад временных компонентов. Их параметры берутся у полной модели,
но число эпох выбирается отдельно на проверочной части.
""")
code("""
classes = {'knn': ItemKNN, 'als': ExplicitALS, 'ease': EASE, 'fm': FM, 'timesvd': TimeSVDPP}
display(pd.DataFrame([{'Модель': name, 'Файл': str(Path(inspect.getfile(cls)).relative_to(ROOT))}
                      for name, cls in classes.items()]))
search_space = pd.DataFrame([
    ['ItemKNN', 'k: 20,50,100,200. gamma: 0,10,50,100', 16],
    ['EASE', 'lambda: 1,3,10,30,100,300,1000,3000', 8],
    ['ALS', 'Размерность 16/32/64. L2 от 0.01 до 100 по логарифмической шкале. До 40 итераций', 16],
    ['FM', 'Размерность 16/32/64. Шаг от 1e-4 до 1e-2, L2 от 1e-6 до 1e-1. До 50 эпох', 16],
    ['timeSVD++', 'Как FM, плюс временной штраф и интервалы 30/90 дней', 16],
    ['MostPopular', 'Без подбора', 1],
], columns=['Модель', 'Пространство поиска', 'Наборы'])
display(search_space)
print('Точные наборы сохранены в configs/benchmark.json')
print('Начальное значение генератора поиска 2026, обучения при подборе', config['tuning_seed'])
print('Остановка после', config['patience'], 'проверок без улучшения. Основная метрика:', PRIMARY_METRIC)
""")
md("""
Ниже выполняется весь заданный поиск. При наличии сохранённого завершённого запуска проверяются
параметры, данные, исходники и окружение. Несовместимый результат не используется.
Тестовые метрики при подборе не считаются.
""")
code("""
with threadpool_limits(limits=config['threads']):
    experiment = Experiment(DATA, Path('configs/benchmark.json'), RUNS)
    selected = experiment.tune()
""")
code("""
chosen = pd.DataFrame([
    {'Модель': name, 'Параметры': json.dumps(row['parameters'], ensure_ascii=False),
     'Лучшая эпоха': row['best_epoch'], 'NDCG@10 на проверке': row['validation'][PRIMARY_METRIC]}
    for name, row in selected.items()
])
display(chosen)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
for ax, name in zip(axes, ['als', 'fm', 'timesvd']):
    trial = json.loads((RUNS / 'trials' / name / f"{selected[name]['trial']:03d}.json").read_text())
    curve = pd.DataFrame(trial['curve'])
    ax.plot(curve.epoch, curve.ndcg, marker='o', markersize=3)
    ax.axvline(selected[name]['best_epoch'], linestyle=':', color='gray')
    ax.set(title=name, xlabel='Эпоха', ylabel='NDCG@10 на проверке')
plt.tight_layout()
plt.show()
""")
md("""
## 5. Итоговое сравнение при K = 10

Настройки уже выбраны. Каждая случайная модель ещё раз выбирает число эпох для своего начального
значения генератора на проверочной части. Потом обучается на обучающей и проверочной частях вместе.
Тест используется только для итогового сравнения.
""")
code("""
with threadpool_limits(limits=config['threads']):
    experiment.final()
from recsys.report import build_report
build_report(DATA, RUNS, REPORT)
subprocess.run([sys.executable, 'scripts/plot_tradeoff.py', str(REPORT)], check=True)
subprocess.run([sys.executable, 'scripts/verify_report.py', str(REPORT)], check=True)
""")
code("""
summary = pd.read_csv(REPORT / 'summary.csv')
table = summary[['label', 'n_seeds', 'ndcg@10_mean', 'ndcg@10_std', 'recall@10_mean',
                 'precision@10_mean', 'map@10_mean', 'catalog_coverage@10_mean']].copy()
table.columns = ['Модель', 'Запуски', 'NDCG@10', 'Отклонение NDCG', 'Recall@10', 'Precision@10', 'MAP@10', 'Покрытие каталога']
table['Параметры'] = [json.dumps(json.loads((RUNS / 'final' / name / '42/run.json').read_text())['parameters'],
                                ensure_ascii=False) for name in summary.model]
display(table.round(6))
print('Число эпох в параметрах показано для начального значения генератора 42. Для остальных запусков оно выбирается отдельно.')
display(Image(filename=str(REPORT / 'ndcg.png')))
display(Image(filename=str(REPORT / 'quality_vs_time.png')))
display(summary[['label', 'fit_seconds_mean', 'test_seconds_mean', 'peak_rss_mib_mean']].rename(
    columns={'label':'Модель', 'fit_seconds_mean':'Обучение, с', 'test_seconds_mean':'Рекомендации и метрики, с', 'peak_rss_mib_mean':'Пиковая память, МиБ'}).round(3))
""")
code("""
contrasts = json.loads((REPORT / 'contrasts.json').read_text())
comparison = pd.DataFrame(contrasts)[['first','second','difference','ci95_low','ci95_high','n_users','n_bootstrap']].copy()
labels = dict(zip(summary.model, summary.label))
comparison['first'] = comparison['first'].map(labels)
comparison['second'] = comparison['second'].map(labels)
comparison.columns = ['Первая модель','Вторая модель','Разность NDCG','Нижняя граница','Верхняя граница','Пользователи','Повторы']
display(comparison)
display(Markdown('Метрики пользователя сначала усредняются по трём запускам. Затем 2 000 раз одни и те же пользователи выбираются с возвращением для обеих моделей. Интервалы относятся к данному периоду и настройкам. Дополнительные сравнения приведены без поправки на множественные проверки.'))
""")
md("""
### Пример рекомендаций

Пользователь выбран по фиксированному правилу из группы активности, независимо от качества моделей.
Будущий отзыв служит проверкой попадания. Если отзыва нет, товар не обязательно плохой.
""")
code("""
from recsys.core import Model
from recsys.ranking import recommend
example = json.loads((REPORT / 'example_selection.json').read_text())['users'][0]
model = Model.load(RUNS / 'final/ease/42/model.pkl')
items, scores, _ = recommend(model, test_snapshot, np.array([example]), k=10)
truth = set(data.test.loc[(data.test.user == example) & (data.test.rating >= 4), 'item'])
recs = pd.DataFrame({'item':items[0], 'Место':range(1,11), 'Оценка модели':scores[0]})
recs['Попадание'] = recs.item.isin(truth)
print('Внутренний ID пользователя:', example)
display(recs.merge(metadata[['item','asin','title']], on='item')[['Место','asin','title','Оценка модели','Попадание']])
del model
gc.collect()
""")
md("""
## 6. Выводы

Полный отчёт находится в `reports/REPORT.md`, параметры в `reports/selected_parameters.json`.
""")
code("""
s = summary.set_index('model')
main = s.loc[list(classes)]
best = main['ndcg@10_mean'].idxmax()
fm_effect = next(c for c in contrasts if c['first']=='fm' and c['second']=='fm_id')
time_effect = next(c for c in contrasts if c['first']=='timesvd' and c['second']=='svdpp')
params = selected[best]['parameters']
display(Markdown(f'''
Лучший NDCG@10 среди пяти методов у {s.loc[best, 'label']}, значение {s.loc[best, 'ndcg@10_mean']:.5f}.
Recall@10 у этой модели {s.loc[best, 'recall@10_mean']:.5f}. Её параметры: `{json.dumps(params)}`.
По Precision@10 выше {s['precision@10_mean'].idxmax()}, по MAP@10 выше {s['map@10_mean'].idxmax()}.

Покрытие каталога у ItemKNN {s.loc['knn','catalog_coverage@10_mean']:.2%},
у EASE {s.loc['ease','catalog_coverage@10_mean']:.2%}, у FM {s.loc['fm','catalog_coverage@10_mean']:.2%}.
У FM рекомендации сосредоточены на небольшом числе товаров.

Метапризнаки FM изменили NDCG на {fm_effect['difference']:.6f}.
95% интервал от {fm_effect['ci95_low']:.6f} до {fm_effect['ci95_high']:.6f}.
Временные компоненты изменили NDCG относительно SVD++ без времени на {time_effect['difference']:.6f},
интервал от {time_effect['ci95_low']:.6f} до {time_effect['ci95_high']:.6f}.
Параметры упрощённых версий унаследованы от полных, число эпох подобрано отдельно.
'''))
""")
md("""
У выбранных ItemKNN и EASE параметры находятся на верхней границе сетки. Поиск за этими границами не проводился.
ALS, FM и timeSVD++ предсказывают наблюдаемые оценки, а не напрямую порядок первых десяти товаров.
Поэтому низкая ошибка оценок сама по себе не гарантирует лучший NDCG.

Ограничения сравнения:

- Это активные пользователи и фиксированный каталог. На новых пользователях и товарах качество не проверялось.
- Только 57,37% будущих положительных событий выбранных пользователей относятся к доступному каталогу.
- Отзывы не дают полного журнала показов. По этим данным нельзя оценить изменение кликов или продаж.
- Поздние названия и категории используются для отображения. В модели не передаются признаки из будущего.
- Время обучения зависит от выбранного числа эпох. Пиковая память относится ко всему процессу.
- Один период и конечный поиск параметров ограничивают выводы о моделях в целом.

### Источники

[Amazon Reviews 2018](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/),
[Item-based CF](https://www.ra.ethz.ch/CDstore/www10/papers/519/),
[EASE](https://arxiv.org/abs/1905.03375), [FM](https://www.libfm.org/),
[timeSVD++](https://cseweb.ucsd.edu/classes/fa17/cse291-b/reading/p447-koren.pdf),
[полное ранжирование каталога](https://research.google/pubs/on-sampled-metrics-for-item-recommendation/).
""")

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata.kernelspec = {'display_name':'Python 3', 'language':'python', 'name':'python3'}
notebook.metadata.language_info = {'name':'python', 'version':'3.12'}
nbf.write(notebook, root / 'experiments.ipynb')
print(f'Создано {len(cells)} ячеек')
