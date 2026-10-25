"""Auditable Russian report from completed, checksum-verified experiment runs.

This module reads scores and metrics only. It never trains a model, selects a
trial on test data, or deserializes a checkpoint. Public user IDs are the frozen
integer IDs; Amazon reviewer IDs and mapping files are not exported.
"""
from __future__ import annotations

import html
import itertools
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .data import load_dataset
from .io import object_hash, sha256, write_json
from .metrics import evaluate, paired_bootstrap


LABELS = {"popular": "MostPopular", "knn": "ItemKNN", "als": "ALS",
          "ease": "EASE", "fm": "FM с признаками", "timesvd": "timeSVD++",
          "fm_id": "FM только ID", "svdpp": "SVD++ без времени"}
ORDER = tuple(LABELS)
DETERMINISTIC = {"popular", "knn", "ease"}
REQUIRED_MODELS = {"knn", "als", "ease", "fm", "timesvd"}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _csv(frame, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def _table(headers, rows):
    def cell(value):
        return html.escape(str(value), quote=False).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(cell, headers)) + " |",
                       "| " + " | ".join(["---"] * len(headers)) + " |"]
                      + ["| " + " | ".join(map(cell, row)) + " |" for row in rows])


def _expected(config):
    models = list(config["search"])
    if config.get("ablations", True):
        models += [name for name, parent in [("fm_id", "fm"), ("svdpp", "timesvd")]
                   if parent in models]
    unknown = set(models) - set(LABELS)
    if unknown:
        raise ValueError(f"Unknown report models: {sorted(unknown)}")
    return [(model, int(seed)) for model in ORDER if model in models
            for seed in ([config["tuning_seed"]] if model in DETERMINISTIC else config["seeds"])]


def _load_runs(data, runs_dir, experiment):
    """Require every configured run, verify bytes, then reconstruct main metrics."""
    config, fingerprint = experiment["config"], experiment["fingerprint"]
    expected = _expected(config)
    missing = [str(runs_dir / "final" / model / str(seed) / "run.json")
               for model, seed in expected
               if not (runs_dir / "final" / model / str(seed) / "run.json").is_file()]
    if missing:
        raise FileNotFoundError("Incomplete experiment; missing final runs: " + ", ".join(missing))
    result = []
    eligible = data.evaluation_users("test", include_outside=True)
    seen = data.snapshot("test").matrix()
    k = int(config["k"])
    for model, seed in expected:
        directory = runs_dir / "final" / model / str(seed)
        run = _read(directory / "run.json")
        if (run.get("status") != "complete" or run.get("fingerprint") != fingerprint
                or run.get("model") != model or run.get("seed") != seed):
            raise ValueError(f"Run identity/status/fingerprint mismatch: {directory}")
        if not {"model.pkl", "per_user.csv", "recommendations.csv"}.issubset(run["files"]):
            raise ValueError(f"Incomplete artifact hashes: {directory}")
        for filename, digest in run["files"].items():
            if Path(filename).name != filename:
                raise ValueError("Artifact filenames must be local basenames")
            if sha256(directory / filename) != digest:
                raise ValueError(f"Artifact checksum mismatch: {directory / filename}")
        per_user = pd.read_csv(directory / "per_user.csv").sort_values("user").reset_index(drop=True)
        recs = pd.read_csv(directory / "recommendations.csv").sort_values(["user", "rank"]).reset_index(drop=True)
        if not np.array_equal(per_user.user.to_numpy(), eligible):
            raise ValueError(f"Different evaluation cohort: {model}/{seed}")
        if (len(recs) != len(eligible) * k
                or not np.array_equal(recs.user.to_numpy(), np.repeat(eligible, k))
                or not np.array_equal(recs["rank"].to_numpy(), np.tile(np.arange(1, k + 1), len(eligible)))
                or not np.isfinite(recs.score).all()):
            raise ValueError(f"Invalid recommendation shape/order/scores: {model}/{seed}")
        matrix = recs.item.to_numpy().reshape(len(eligible), k)
        for user, items in zip(eligible, matrix):
            if np.intersect1d(seen.indices[seen.indptr[user]:seen.indptr[user + 1]], items).size:
                raise ValueError(f"Seen item in recommendations: {model}/{seed}")
        rebuilt, metrics = evaluate(matrix, eligible, data.test, len(data.items), k=k,
                                    allfuture_counts=data.allfuture_counts["test"])
        for column in rebuilt.columns:
            np.testing.assert_allclose(per_user[column], rebuilt[column], rtol=1e-9, atol=1e-12,
                                       equal_nan=True, err_msg=f"Per-user metric mismatch: {model}/{seed}/{column}")
        for metric, value in metrics.items():
            if not np.isclose(run["metrics"][metric], value, rtol=1e-9, atol=1e-12, equal_nan=True):
                raise ValueError(f"Aggregate metric mismatch: {model}/{seed}/{metric}")
        result.append({"model": model, "seed": seed, "directory": directory,
                       "run": run, "per_user": per_user, "recommendations": recs})
    return result


def _summary(runs):
    records = []
    for record in runs:
        run = record["run"]
        records.append({"model": record["model"], "seed": record["seed"],
                        **run["metrics"], **{key: run[key] for key in
                                            ["fit_seconds", "test_seconds", "peak_rss_mib"]}})
    raw = pd.DataFrame(records)
    rows = []
    columns = [key for key in raw if key not in {"model", "seed"}]
    for model in ORDER:
        selected = raw[raw.model == model]
        if selected.empty:
            continue
        row = {"model": model, "label": LABELS[model], "n_seeds": len(selected)}
        for key in columns:
            values = selected[key].dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            row[f"{key}_mean"] = float(values.mean())
            row[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows), raw


def _contrasts(runs, k):
    by_model = {model: pd.concat([r["per_user"] for r in runs if r["model"] == model],
                                ignore_index=True) for model in {r["model"] for r in runs}}
    results = []
    for left, right, role in [("timesvd", "als", "primary"),
                              ("timesvd", "svdpp", "exploratory"),
                              ("fm", "fm_id", "exploratory")]:
        if left in by_model and right in by_model:
            interval = paired_bootstrap(by_model[left], by_model[right], metric=f"ndcg@{k}",
                                         n_bootstrap=2000, seed=2026)
            results.append({"first": left, "second": right, "role": role, **interval})
    return results


def _activity_and_overlap(data, runs, k):
    activity = data.train.groupby("user").size()
    strata_rows = []
    for model in ORDER:
        records = [r for r in runs if r["model"] == model]
        if not records:
            continue
        users = pd.concat([r["per_user"] for r in records]).groupby("user")[
            [f"ndcg@{k}", f"recall@{k}"]].mean().dropna()
        counts = activity.reindex(users.index).to_numpy()
        strata = np.where(counts <= 9, "5–9", np.where(counts <= 19, "10–19", "20+"))
        for stratum in ["5–9", "10–19", "20+"]:
            subset = users.loc[strata == stratum]
            strata_rows.append({"model": model, "train_history": stratum, "n_users": len(subset),
                                **{column: float(subset[column].mean()) for column in subset}})
    # Qualitative/list comparisons use seed 42, never the best test seed.
    comparisons = {r["model"]: r["recommendations"] for r in runs if r["seed"] == 42}
    eligible = set(data.evaluation_users("test").tolist())
    overlaps = []
    for first, second in itertools.combinations(comparisons, 2):
        a = {int(u): set(g.item) for u, g in comparisons[first].groupby("user") if u in eligible}
        b = {int(u): set(g.item) for u, g in comparisons[second].groupby("user") if u in eligible}
        common = sorted(set(a) & set(b))
        overlaps.append({"first": first, "second": second, "seed": 42, "n_users": len(common),
                         f"mean_overlap@{k}": float(np.mean([len(a[u] & b[u]) / k for u in common])),
                         "mean_jaccard": float(np.mean([len(a[u] & b[u]) / len(a[u] | b[u]) for u in common]))})
    return pd.DataFrame(strata_rows), pd.DataFrame(overlaps)


def _example_users(data):
    """One RNG-2026 choice in each activity third; no recommendation scores used."""
    eligible = data.evaluation_users("test")
    counts = data.train.groupby("user").size().reindex(eligible).to_numpy()
    ordered = eligible[np.lexsort((eligible, counts))]
    rng = np.random.default_rng(2026)
    return [int(rng.choice(group)) for group in np.array_split(ordered, 3) if len(group)]


def _examples(data, data_dir, runs, k):
    titles = {}
    metadata_path = data_dir / "metadata.csv"
    if metadata_path.exists():
        manifest = _read(data_dir / "metadata_manifest.json")
        if sha256(metadata_path) != manifest["file_sha256"]:
            raise ValueError("Metadata checksum mismatch")
        metadata = pd.read_csv(metadata_path).fillna("")
        if not np.array_equal(metadata.item, np.arange(len(data.items))) or metadata.asin.tolist() != data.items:
            raise ValueError("Metadata do not match the frozen item mapping")
        titles = {int(row.item): str(row.title) for row in metadata.itertuples() if str(row.title).strip()}
    def label(item):
        # Strip line breaks/HTML from catalog text; no reviewer identifiers are used.
        title = " ".join(titles.get(int(item), "").split())
        return f"{title[:140]} [{data.items[int(item)]}]" if title else data.items[int(item)]
    selected = _example_users(data)
    history = pd.concat([data.train, data.validation], ignore_index=True)
    lines = ["# Примеры персональных рекомендаций", "",
             "Из трёх групп активности в обучающей части выбрано по одному пользователю. Начальное "
             "значение генератора при выборе 2026, при обучении показанных моделей 42. Качество моделей "
             "при выборе не учитывалось. Пользователи обозначены внутренними ID, названия взяты из каталога. "
             "Если положительного тестового отзыва нет, релевантность товара неизвестна.", ""]
    available = [r for r in runs if r["seed"] == 42]
    if not available:
        lines += ["Нет запуска с начальным значением генератора 42, поэтому примеры не выбраны.", ""]
    for user in selected:
        past = history[history.user == user].sort_values(["timestamp", "item"]).tail(5)
        truth = set(data.test.loc[(data.test.user == user) & (data.test.rating >= 4), "item"])
        lines += [f"## Пользователь {user}", "", "Последние пять оценок до начала тестового периода:", "",
                  _table(["Дата UTC", "Товар", "Оценка"],
                         [(pd.to_datetime(r.timestamp, unit="s", utc=True).strftime("%Y-%m-%d"),
                           label(r.item), int(r.rating)) for r in past.itertuples()]), ""]
        for record in available:
            recs = record["recommendations"].loc[record["recommendations"].user == user]
            lines += [f"### {LABELS[record['model']]}", "",
                      _table(["Место", "Товар", "Есть положительная оценка в тесте"],
                             [(r.rank, label(r.item), "да" if r.item in truth else "нет")
                              for r in recs.itertuples()]), ""]
    return "\n".join(lines), selected


def _plots(summary, output, k):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    positions = np.arange(len(summary))
    colors = ["#6b7280" if m == "popular" else "#a5b4fc" if m in {"fm_id", "svdpp"}
              else "#3568a8" for m in summary.model]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(positions, summary[f"ndcg@{k}_mean"], yerr=summary[f"ndcg@{k}_std"],
           color=colors, capsize=4)
    ax.set_xticks(positions, summary.label, rotation=25, ha="right")
    ax.set_ylabel(f"NDCG@{k}")
    ax.set_title("Качество рекомендаций и разброс между запусками")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output / "ndcg.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    for (_, row), color in zip(summary.iterrows(), colors):
        ax.errorbar(row.fit_seconds_mean, row[f"ndcg@{k}_mean"],
                    xerr=row.fit_seconds_std, yerr=row[f"ndcg@{k}_std"],
                    fmt="o", color=color, capsize=3)
        ax.annotate(row.label, (row.fit_seconds_mean, row[f"ndcg@{k}_mean"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("Время обучения, с (симметричная логарифмическая шкала)")
    ax.set_ylabel(f"NDCG@{k}")
    ax.set_title("Качество и время обучения")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "quality_vs_time.png", dpi=180)
    plt.close(fig)


def _audit_export(runs_dir, output, experiment, runs):
    """Copy compact evidence; never copy checkpoints, users.json or raw reviews."""
    audit = output / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    write_json(audit / "experiment.json", experiment)
    copied = []
    for record in runs:
        relative = Path("final") / record["model"] / str(record["seed"])
        destination = audit / relative
        destination.mkdir(parents=True, exist_ok=True)
        for filename in ["run.json", "per_user.csv", "recommendations.csv"]:
            shutil.copyfile(record["directory"] / filename, destination / filename)
            copied.append((relative / filename).as_posix())
    for section in ["selected", "trials", "seed_validation", "pilot"]:
        for source in sorted((runs_dir / section).rglob("*.json")):
            if source.name.endswith(".progress.json"):
                continue
            contents = _read(source)
            if section != "selected" and contents.get("fingerprint") != experiment["fingerprint"]:
                raise ValueError(f"Mixed experiment fingerprints: {source}")
            relative = source.relative_to(runs_dir)
            destination = audit / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            copied.append(relative.as_posix())
    copied.append("experiment.json")
    write_json(audit / "manifest.json", {
        "experiment_fingerprint": experiment["fingerprint"],
        "checkpoint_exported": False,
        "note": "run.json preserves source checkpoint hashes; model.pkl is not included. "
                "User IDs are local integer IDs; raw reviewer identifiers and mappings are omitted.",
        "files": {filename: sha256(audit / filename) for filename in sorted(copied)}})


def build_report(data_dir, runs_dir, output_dir) -> None:
    """Build tables, report, plots and compact evidence only for completed runs."""
    data_dir, runs_dir, output = Path(data_dir), Path(runs_dir), Path(output_dir)
    data = load_dataset(data_dir)
    experiment = _read(runs_dir / "experiment.json")
    previous_report = output / "report_manifest.json"
    if previous_report.exists() and _read(previous_report)["experiment_fingerprint"] != experiment["fingerprint"]:
        raise ValueError("Report directory belongs to a different experiment; use a new output directory")
    if object_hash(data.manifest) != object_hash(experiment["data"]):
        raise ValueError("Prepared data differ from the experiment manifest")
    runs = _load_runs(data, runs_dir, experiment)
    config = experiment["config"]
    k = int(config["k"])
    selected = {}
    trial_rows = []
    for model, parameters in config["search"].items():
        selected[model] = _read(runs_dir / "selected" / f"{model}.json")
        for index, params in enumerate(parameters):
            trial = _read(runs_dir / "trials" / model / f"{index:03d}.json")
            if (trial.get("fingerprint") != experiment["fingerprint"]
                    or trial.get("status") != "complete" or trial.get("parameters") != params):
                raise ValueError(f"Missing or inconsistent validation trial: {model}/{index}")
            trial_rows.append({"model": model, "trial": index, "best_epoch": trial["best_epoch"],
                               f"validation_ndcg@{k}": trial["validation"][f"ndcg@{k}"],
                               "fit_seconds": trial["fit_seconds"]})
        best = max(range(len(parameters)), key=lambda i: trial_rows[-len(parameters) + i][f"validation_ndcg@{k}"])
        if selected[model]["trial"] != best or selected[model]["parameters"] != parameters[best]:
            raise ValueError(f"Selected parameters do not match validation winner: {model}")
    output.mkdir(parents=True, exist_ok=True)
    summary, raw_summary = _summary(runs)
    _csv(summary, output / "summary.csv")
    _csv(raw_summary, output / "runs.csv")
    _csv(pd.DataFrame(trial_rows), output / "validation_trials.csv")
    write_json(output / "selected_parameters.json", selected)
    contrasts = _contrasts(runs, k)
    write_json(output / "contrasts.json", contrasts)
    activity, overlaps = _activity_and_overlap(data, runs, k)
    _csv(activity, output / "activity_strata.csv")
    _csv(overlaps, output / "list_overlap.csv")
    examples, example_users = _examples(data, data_dir, runs, k)
    (output / "EXAMPLES.md").write_text(examples + "\n", encoding="utf-8")
    write_json(output / "example_selection.json", {"seed": 2026, "users": example_users,
               "rule": "One user sampled per activity third, ordered by train interaction count then local user ID; test-positive cohort; independent of model scores."})
    _plots(summary, output, k)
    _audit_export(runs_dir, output, experiment, runs)
    write_json(output / "data_manifest.json", data.manifest)
    write_json(output / "provenance.json", {"experiment_fingerprint": experiment["fingerprint"],
               "environment": experiment["environment"], "config": config,
               "source": data.manifest["source"]})
    metadata_manifest = data_dir / "metadata_manifest.json"
    if metadata_manifest.exists():
        shutil.copyfile(metadata_manifest, output / "metadata_manifest.json")

    def measure(row, metric, digits=5):
        return f"{row[f'{metric}_mean']:.{digits}f} ± {row[f'{metric}_std']:.{digits}f}"
    main = summary[summary.model.isin(REQUIRED_MODELS)]
    complete = REQUIRED_MODELS.issubset(set(summary.model))
    winner = main.sort_values(f"ndcg@{k}_mean", ascending=False, kind="stable").iloc[0] if len(main) else None
    validation_date = pd.to_datetime(data.validation_start, unit="s", utc=True).strftime("%Y-%m-%d")
    test_date = pd.to_datetime(data.test_start, unit="s", utc=True).strftime("%Y-%m-%d")
    n_test = len(data.evaluation_users("test"))
    micro_coverage = data.manifest["exclusions"]["test"]["positive_event_catalog_coverage"]
    lines = ["# Сравнение рекомендаций Amazon Musical Instruments 2018", "",
             ("ItemKNN, ALS для явных оценок, EASE, FM с метапризнаками и timeSVD++. "
              "Для сравнения добавлены MostPopular, FM только с ID и SVD++ без времени."
              if complete else "Результаты доступных моделей приведены в таблице."), "",
             "## Данные и протокол", "",
             f"После очистки: {data.manifest['total_interactions']} взаимодействий, {len(data.users)} "
             f"пользователей и {len(data.items)} товаров. Из обучающей части по очереди удаляются "
             f"пользователи и товары с менее чем {data.manifest['protocol']['core_k']} взаимодействиями. "
             "Для повторной пары пользователь-товар остаётся первое событие.", "",
             _table(["Часть", "Взаимодействия", "Пользователи с положительной оценкой"],
                    [({"train": "Обучение", "validation": "Проверка", "test": "Тест"}[name], data.manifest["splits"][name]["rows"],
                      data.manifest["splits"][name]["positive_users"]) for name in ["train", "validation", "test"]]), "",
             f"Обучение до {validation_date}, проверка от {validation_date} до {test_date}, "
             f"тест с {test_date}. Даты в UTC. События одного дня остаются вместе. Параметры и число эпох "
             "подбираются на проверочной части, затем модель обучается на двух частях вместе.", "",
             f"Основные метрики считаются по {n_test} пользователям с будущей оценкой ≥ 4 внутри каталога. "
             f"Для каждого ранжируется весь каталог из {len(data.items)} товаров за вычетом всех уже оценённых "
             f"товаров, включая оценки 1-3. Выбирается {k} рекомендаций. При равных оценках первым идёт "
             "меньший ID товара. Если персонального сигнала нет, используются популярные товары.", "",
             "## Результаты", "",
             "Среднее ± стандартное отклонение запусков. Модели без случайной инициализации запущены "
             "один раз, для них отклонение записано нулём.", "",
             _table(["Модель", "Запуски", f"NDCG@{k}", f"Recall@{k}", f"Precision@{k}", f"MAP@{k}"],
                    [(row.label, row.n_seeds, measure(row, f"ndcg@{k}"), measure(row, f"recall@{k}"),
                      measure(row, f"precision@{k}"), measure(row, f"map@{k}")) for _, row in summary.iterrows()]), ""]
    if winner is not None:
        lines += [f"Лучшее среднее NDCG@{k} на тесте у {winner.label}: "
                  f"{winner[f'ndcg@{k}_mean']:.5f}.", ""]
    lines += ["![Качество рекомендаций](ndcg.png)", "", "## Парные сравнения", "",
              "Основное сравнение: timeSVD++ минус ALS. Метрики каждого пользователя усредняются по "
              "запускам. Затем 2000 раз одни и те же пользователи выбираются с возвращением для обеих "
              "моделей, начальное значение генератора 2026. Границы 95% интервала берутся по процентилям "
              "разностей. Упрощённые версии получают параметры полной модели и отдельно выбирают число "
              "эпох на проверочной части. Эти сравнения дополнительные.", "",
              _table(["Разность", "Сравнение", f"Δ NDCG@{k}", "95% интервал"],
                     [(f"{LABELS[c['first']]} − {LABELS[c['second']]}",
                       {"primary": "основное", "exploratory": "дополнительное"}[c["role"]], f"{c['difference']:.5f}",
                       f"[{c['ci95_low']:.5f}, {c['ci95_high']:.5f}]") for c in contrasts]), "",
              "У timeSVD++ и ALS различаются и учёт времени, и использование неявной истории. "
              "Версия SVD++ без времени нужна для проверки временных компонентов.", "",
              "## Вычислительная стоимость", "",
              _table(["Модель", "Обучение, с", "Рекомендации и метрики, с", "Пиковая память, МиБ"],
                     [(row.label, measure(row, "fit_seconds", 3), measure(row, "test_seconds", 3),
                       measure(row, "peak_rss_mib", 1)) for _, row in summary.iterrows()]), "",
              "Время измерено для одного итогового обучения. Память охватывает весь процесс, вместе "
              "с библиотеками и данными, и измеряется каждые 50 мс. Окружение и число потоков "
              "записаны в provenance.json.", "", "![Качество и время](quality_vs_time.png)", "",
              "## Формулы метрик", "",
              f"Для пользователя u: Rᵤ содержит положительные тестовые товары. hⱼ = 1, если рекомендация "
              f"на позиции j входит в Rᵤ. Precision@{k} = Σhⱼ/{k}. Recall@{k} = Σhⱼ/|Rᵤ|. "
              f"AP@{k} = Σⱼ hⱼ·(Σₗ≤ⱼhₗ/j) / min({k}, |Rᵤ|). "
              "DCG = Σⱼ hⱼ/log₂(j+1), NDCG = DCG/IDCG с идеальным размещением релевантных товаров. "
              "MAP и остальные основные метрики усредняются по пользователям с непустым Rᵤ.", "",
              "## Условия оценки", "",
              "Результаты относятся к активным пользователям и фиксированному каталогу. "
              "Новые пользователи и товары, а также эффект на реальных показах не проверялись.", "",
              f"Каталог охватывает {micro_coverage:.2%} будущих положительных событий выбранных пользователей. "
              f"Recall_all_future@{k} учитывает товары вне каталога в знаменателе и дополнительно включает "
              "пользователей с положительными оценками только вне каталога.", "",
              "FM использует прошлые количества и средние оценки, возраст истории и календарные признаки. "
              "В исторические признаки и историю timeSVD++ входят только предыдущие дни. "
              "Поправки для неизвестных пар пользователь-день и товар-период равны нулю. Названия из каталога "
              "используются для отображения рекомендаций.", "",
              "summary.csv содержит покрытие каталога, популярность и долю замен на популярные товары. "
              "activity_strata.csv показывает качество по числу обучающих взаимодействий (5-9, 10-19, 20+). "
              f"list_overlap.csv содержит пересечение top-{k} и коэффициент Жаккара для начального значения генератора 42. "
              "[Примеры рекомендаций](EXAMPLES.md) выбраны независимо от качества моделей.", "",
              "## Артефакты", "",
              f"Идентификатор эксперимента: `{experiment['fingerprint']}`. В audit/ сохранены журналы, "
              f"рекомендации и метрики по пользователям для {len(runs)} итоговых запусков. "
              "audit/manifest.json содержит их SHA256. Пользователи обозначены внутренними ID. "
              "Хеши model.pkl в run.json относятся к локальным весам.", "",
              "Выбранные параметры: selected_parameters.json. Конфигурация и окружение: provenance.json. "
              "Правила подготовки и состав выборки: data_manifest.json.", "",
              "## Источники", "",
              "- [Amazon Reviews 2018, официальный источник](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)",
              "- [EASE: Embarrassingly Shallow Autoencoders for Sparse Data](https://arxiv.org/abs/1905.03375)",
              "- [Factorization Machines, авторский сайт](https://www.libfm.org/)",
              "- [Collaborative Filtering with Temporal Dynamics](https://cseweb.ucsd.edu/classes/fa17/cse291-b/reading/p447-koren.pdf)",
              "- [On Sampled Metrics for Item Recommendation](https://research.google/pubs/on-sampled-metrics-for-item-recommendation/)", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(output / "report_manifest.json", {
        "status": "complete", "all_five_models": complete,
        "experiment_fingerprint": experiment["fingerprint"], "n_final_runs": len(runs),
        "files": {p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob("*"))
                  if p.is_file() and p.name != "report_manifest.json"}})
