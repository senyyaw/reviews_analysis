"""Driver analysis: which product aspects move the star rating?

This follows the Artefact approach ("Using NLP to extract quick and valuable insights from your
customers' reviews"): fit an interpretable classifier of LOW (1-2 stars) vs HIGH (4-5 stars)
reviews on aspect features, with generic sentiment words removed, and read its coefficients.
3-star reviews are left out, so the contrast is clean.

Rigour beyond the blog post:
- L2 logistic regression with C chosen by 5-fold cross-validation (ROC-AUC). The CV AUC is
  reported, so a reader can see whether the features predict ratings at all.
- Bootstrap stability: refit on 200 resamples. A driver is reported only when its coefficient has
  the same sign in >= 90% of them, and its median coefficient and odds ratio are given.
- Features: binary aspect n-grams (keywords.doc_terms) with document frequency >= MIN_DF.
A positive coefficient means mentioning the aspect raises the odds of a LOW rating.
"""

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold

MIN_DF = 5
N_BOOT = 200
STABILITY = 0.90


@dataclass(frozen=True)
class Driver:
    term: str
    coef: float  # median over bootstraps; > 0 pushes towards a LOW rating
    odds_ratio: float  # exp(coef)
    stability: float  # share of bootstraps with the same sign as the median
    df: int  # number of reviews mentioning the term


@dataclass(frozen=True)
class DriverResult:
    drivers: list[Driver]
    cv_auc: float
    c: float
    n_low: int
    n_high: int
    n_features: int


def driver_analysis(docs: Sequence[set[str]], low: Sequence[bool], seed: int = 0) -> DriverResult:
    df: dict[str, int] = {}
    for d in docs:
        for t in d:
            df[t] = df.get(t, 0) + 1
    vocab = sorted(t for t, n in df.items() if n >= MIN_DF)
    vec = DictVectorizer(sparse=True)
    vec.fit([{t: 1 for t in vocab}])
    x = vec.transform([{t: 1 for t in d if t in df and df[t] >= MIN_DF} for d in docs])
    y = np.array(low, dtype=int)

    with warnings.catch_warnings():  # sklearn 1.9 deprecation notices about future defaults
        warnings.simplefilter("ignore", FutureWarning)
        cv = _fit_cv(x, y, seed)
    c = float(cv.C_[0])
    cv_auc = float(np.max(np.mean(cv.scores_[1], axis=0)))
    return _bootstrap(x, y, c, cv_auc, vec, df, seed)


def _fit_cv(x: object, y: np.ndarray, seed: int) -> LogisticRegressionCV:
    return LogisticRegressionCV(
        Cs=np.logspace(-2, 1, 10),
        cv=StratifiedKFold(5, shuffle=True, random_state=seed),
        scoring="roc_auc",
        max_iter=5000,
    ).fit(x, y)


def _bootstrap(
    x: object,
    y: np.ndarray,
    c: float,
    cv_auc: float,
    vec: DictVectorizer,
    df: dict[str, int],
    seed: int,
) -> DriverResult:
    rng = np.random.default_rng(seed)
    n_rows, n_cols = x.shape  # type: ignore[attr-defined]
    coefs = np.zeros((N_BOOT, n_cols))
    for b in range(N_BOOT):
        idx = rng.integers(0, n_rows, n_rows)
        coefs[b] = LogisticRegression(C=c, max_iter=5000).fit(x[idx], y[idx]).coef_[0]  # type: ignore[index]
    median = np.median(coefs, axis=0)
    stability = np.mean(np.sign(coefs) == np.sign(median), axis=0)
    names = vec.get_feature_names_out()
    drivers = [
        Driver(
            str(names[i]),
            float(median[i]),
            float(np.exp(median[i])),
            float(stability[i]),
            df[str(names[i])],
        )
        for i in range(len(names))
        if stability[i] >= STABILITY and median[i] != 0
    ]
    return DriverResult(
        drivers=sorted(drivers, key=lambda d: -d.coef),
        cv_auc=cv_auc,
        c=c,
        n_low=int(y.sum()),
        n_high=int((1 - y).sum()),
        n_features=len(names),
    )
