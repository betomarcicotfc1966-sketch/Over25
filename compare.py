"""compare.py : compare votre modèle (model.py) aux marchés et au Dixon-Coles de penaltyblog.

À placer dans le même dossier que model.py. Usage :
    pip install penaltyblog
    python compare.py --divs E0 SP1 I1 D1 F1 --seasons 5 --test-seasons 2

Méthode : backtest walk-forward (le modèle n'utilise que le passé). Les réglages
(calibrage, poids du modèle) sont choisis sur la 1re moitié de la période testée,
puis TOUT est évalué sur la 2e moitié, jamais vue : pas d'optimisme caché.
Tous les prédicteurs sont comparés sur exactement les mêmes matchs.
"""
import argparse
import warnings

import numpy as np
import pandas as pd

import model as M

warnings.filterwarnings("ignore")


def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def logloss(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def penaltyblog_over25(hist, bt, test_start, step_days, half_life=M.HALF_LIFE):
    """Probabilité d'Over 2.5 du Dixon-Coles de penaltyblog, mêmes blocs de 30 jours que M.backtest."""
    import penaltyblog as pb

    out = pd.Series(np.nan, index=bt.index)
    t0 = pd.Timestamp(test_start)
    block = ((bt.Date - t0).dt.days // step_days).astype(int)
    for (div, b), test in bt.groupby(["Div", block]):
        a = t0 + pd.Timedelta(days=int(b) * step_days)
        train = hist[(hist.Div == div) & (hist.Date < a)]
        if len(train) < 150:
            continue
        w = 0.5 ** ((a - train.Date).dt.days.clip(lower=0).values / half_life)
        try:
            clf = pb.models.DixonColesGoalModel(train.FTHG.values, train.FTAG.values,
                                                train.HomeTeam.values, train.AwayTeam.values, w)
            clf.fit()
        except Exception as e:
            print(f"  penaltyblog : échec ajustement {div} {a:%Y-%m-%d} ({e})")
            continue
        for idx, r in test.iterrows():
            try:
                grid = clf.predict(r.HomeTeam, r.AwayTeam)
                out[idx] = float(grid.total_goals("over", 2.5))
            except Exception:
                pass  # équipe inconnue du jeu d'entraînement : exclue de l'échantillon commun
    return out


def verdict(diff):
    diff = pd.Series(diff).dropna()
    if len(diff) < 100:
        return "échantillon trop petit"
    m, se = diff.mean(), diff.std(ddof=1) / np.sqrt(len(diff))
    return "MEILLEUR" if m < -2 * se else "MOINS BON" if m > 2 * se else "indiscernable"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--divs", nargs="+", default=["E0", "SP1", "I1", "D1", "F1"])
    ap.add_argument("--seasons", type=int, default=5, help="saisons de données chargées")
    ap.add_argument("--test-seasons", type=int, default=2, help="saisons rejouées en backtest")
    ap.add_argument("--step", type=int, default=30, help="taille des blocs walk-forward (jours)")
    ap.add_argument("--no-pb", action="store_true", help="ne pas utiliser penaltyblog")
    ap.add_argument("--out", default="comparison.csv")
    a = ap.parse_args()

    print("Chargement des données (football-data.co.uk)…")
    failed = []
    df = M.load_results(a.divs, a.seasons, failed=failed)
    hist, _ = M.add_elo(df)
    if failed:
        print(f"  {len(failed)} fichier(s) indisponible(s) : {', '.join(failed[:6])}")

    y0 = int(hist.Season.max())
    test_start = f"{y0 - (a.test_seasons - 1)}-08-01"
    print(f"Backtest walk-forward depuis {test_start} (blocs de {a.step} jours)…")
    bt = M.backtest(hist, test_start, step_days=a.step).sort_values("Date")

    if not a.no_pb:
        print("Ajustement Dixon-Coles de penaltyblog (quelques minutes)…")
        try:
            bt["p_pb"] = penaltyblog_over25(hist, bt, test_start, a.step)
        except ImportError:
            print("  penaltyblog absent (pip install penaltyblog) : comparaison sans lui.")
            a.no_pb = True

    # --- réglage sur la 1re moitié, évaluation sur la 2e (jamais vue)
    cut = bt.Date.iloc[len(bt) // 2]
    tune_set, hold = bt[bt.Date < cut].copy(), bt[bt.Date >= cut].copy()
    print(f"Réglage : {len(tune_set)} matchs jusqu'au {cut:%d/%m/%Y} · "
          f"Évaluation : {len(hold)} matchs après cette date.")
    tuned = M.auto_tune(tune_set)
    cal, w = tuned["cal"], tuned["w"]
    print(f"Réglages trouvés sur la 1re moitié : poids du modèle {w:.0%}, "
          f"calibrage {'appliqué' if tuned['cal_applied'] else 'non appliqué'}.")

    h = hold
    h["y"] = ((h.FTHG + h.FTAG) > 2).astype(int)
    h["p_model"] = M.apply_calibration(h.p_over25, cal)
    h["p_mkt"] = M.devig(h.o_over, h.o_under)
    h["p_cl"] = M.devig(h.c_over, h.c_under)
    h["p_mix"] = w * h.p_model + (1 - w) * h.p_mkt
    cols = ["y", "p_over25", "p_model", "p_mkt", "p_cl", "p_mix"]
    if not a.no_pb:
        h["p_pb_mix"] = w * h.p_pb + (1 - w) * h.p_mkt
        cols += ["p_pb", "p_pb_mix"]
    s = h.dropna(subset=cols)
    print(f"Échantillon commun à tous les prédicteurs : {len(s)} matchs.\n")

    base = float(tune_set.assign(y=((tune_set.FTHG + tune_set.FTAG) > 2)).y.mean())
    preds = [("Toujours la moyenne (réf.)", np.full(len(s), base)),
             ("Votre modèle (brut)", s.p_over25.values),
             ("Votre modèle (recalibré)", s.p_model.values),
             ("Votre mix modèle + marché", s.p_mix.values),
             ("Marché avant-match", s.p_mkt.values),
             ("Marché clôture Pinnacle", s.p_cl.values)]
    if not a.no_pb:
        preds += [("penaltyblog Dixon-Coles", s.p_pb.values),
                  ("penaltyblog + marché", s.p_pb_mix.values)]

    ref = (s.p_cl.values - s.y.values) ** 2
    rows = []
    for name, p in preds:
        rows.append({"Prédicteur": name, "Brier": round(brier(s.y, p), 5),
                     "Log loss": round(logloss(s.y, p), 5),
                     "vs clôture": verdict((p - s.y.values) ** 2 - ref) if "clôture" not in name else "—"})
    table = pd.DataFrame(rows).sort_values("Brier").reset_index(drop=True)
    print("=== Qui prédit le mieux ? (plus bas = mieux ; 'vs clôture' = test pairé à 2 écarts-types) ===")
    print(table.to_string(index=False))
    table.to_csv(a.out, index=False)

    print("\n=== Paris simulés sur la période d'évaluation (cotes moyennes, mise fixe) ===")
    _, _, bets = M.evaluate(hold, w_model=w, cal=cal)
    print(bets.to_string(index=False))
    print("\nLecture : un CLV positif et stable sur plusieurs centaines de paris compte bien plus "
          "que le ROI. Si 'Marché clôture' est en tête et que 'vs clôture' dit MOINS BON ou "
          "indiscernable pour votre mix, n'attendez pas d'avantage durable.")
    print(f"\nTableau enregistré dans {a.out}")


if __name__ == "__main__":
    main()
