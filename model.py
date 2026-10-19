"""Moteur Over/Under 2.5 buts.

Modèle : régression de Poisson régularisée (attaque, défense, avantage du terrain),
pondération temporelle (matchs récents comptent plus) et écart d'Élo calculé
match après match (aucune fuite de données du futur). Données : football-data.co.uk (gratuit, sans clé).
"""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import OneHotEncoder

BASE = "https://www.football-data.co.uk"
LEAGUES = {
    "E0": "Premier League", "E1": "Championship", "SP1": "La Liga", "I1": "Serie A",
    "D1": "Bundesliga", "F1": "Ligue 1", "N1": "Eredivisie", "P1": "Primeira Liga",
    "B1": "Pro League", "T1": "Süper Lig",
}
HALF_LIFE = 240      # jours : un match vieux de 240 jours compte moitié moins
ALPHA = 0.002        # régularisation (évite le sur-apprentissage)
ELO_K, ELO_HA, ELO_REGRESS = 20, 65, 0.3

# Colonnes de cotes (ordre = priorité). Pinnacle = bookmaker de référence du marché.
PRE_PAIRS = [("P>2.5", "P<2.5"), ("Avg>2.5", "Avg<2.5"), ("B365>2.5", "B365<2.5")]
BET_PAIRS = [("Avg>2.5", "Avg<2.5"), ("B365>2.5", "B365<2.5"), ("P>2.5", "P<2.5")]
CLOSE_PAIRS = [("PC>2.5", "PC<2.5"), ("AvgC>2.5", "AvgC<2.5")]


# ---------------------------------------------------------------- données
def _read(url):
    df = pd.read_csv(url, encoding="latin-1", on_bad_lines="skip")
    # le fichier commence parfois par un caractère invisible (BOM) qui abîme le nom de la 1re colonne
    df.columns = [str(c).replace("ï»¿", "").replace("\ufeff", "").strip() for c in df.columns]
    return df


def season_codes(n, today=None):
    today = today or pd.Timestamp.today()
    y = today.year if today.month >= 7 else today.year - 1
    return [(f"{(y - i) % 100:02d}{(y - i + 1) % 100:02d}", y - i) for i in range(n)]


def pick_odds(df, pairs):
    o = pd.Series(np.nan, index=df.index, dtype=float)
    u = o.copy()
    for co, cu in pairs:
        if co in df.columns and cu in df.columns:
            a = pd.to_numeric(df[co], errors="coerce")
            b = pd.to_numeric(df[cu], errors="coerce")
            m = o.isna() & (a > 1) & (b > 1)
            o[m], u[m] = a[m], b[m]
    return o, u


def devig(over, under):
    """Retire la marge du bookmaker : probabilité 'juste' de l'Over."""
    io, iu = 1 / over, 1 / under
    return io / (io + iu)


def add_odds(df):
    df["o_over"], df["o_under"] = pick_odds(df, PRE_PAIRS)
    df["b_over"], df["b_under"] = pick_odds(df, BET_PAIRS)
    df["c_over"], df["c_under"] = pick_odds(df, CLOSE_PAIRS)
    return df


def load_results(divs, n_seasons=5, today=None, reader=_read):
    frames = []
    for code, y in season_codes(n_seasons, today):
        for d in divs:
            try:
                df = reader(f"{BASE}/mmz4281/{code}/{d}.csv")
            except Exception:
                continue
            df["Div"], df["Season"] = d, y
            frames.append(df)
    if not frames:
        raise RuntimeError("Aucune donnée téléchargée : vérifiez la connexion internet.")
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce", format="mixed")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    df["FTHG"], df["FTAG"] = df["FTHG"].astype(int), df["FTAG"].astype(int)
    return add_odds(df).sort_values("Date").reset_index(drop=True)


def load_fixtures(divs, reader=_read):
    fx = reader(f"{BASE}/fixtures.csv")
    fx = fx[fx["Div"].isin(divs)].copy()
    fx["Date"] = pd.to_datetime(fx["Date"], dayfirst=True, errors="coerce", format="mixed")
    fx = fx.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
    return add_odds(fx)


# ---------------------------------------------------------------- Élo
def add_elo(df):
    """Ajoute elo_diff (Élo domicile - extérieur AVANT le match). Renvoie (df, Élo actuel)."""
    E, seen = {}, {}
    div, sea = df["Div"].values, df["Season"].values
    H, A = df["HomeTeam"].values, df["AwayTeam"].values
    hg, ag = df["FTHG"].values, df["FTAG"].values
    diff = np.empty(len(df))
    for i in range(len(df)):
        d = div[i]
        if seen.get(d) != sea[i]:
            if d in seen:  # nouvelle saison : retour partiel vers la moyenne
                for k in [k for k in E if k[0] == d]:
                    E[k] = 1500 + (1 - ELO_REGRESS) * (E[k] - 1500)
            seen[d] = sea[i]
        kh, ka = (d, H[i]), (d, A[i])
        eh, ea = E.get(kh, 1500.0), E.get(ka, 1500.0)
        diff[i] = eh - ea
        we = 1 / (1 + 10 ** (-(eh - ea + ELO_HA) / 400))
        w = 1.0 if hg[i] > ag[i] else 0.5 if hg[i] == ag[i] else 0.0
        n = abs(hg[i] - ag[i])
        g = 1.0 if n <= 1 else 1.5 if n == 2 else (11 + n) / 8
        delta = ELO_K * g * (w - we)
        E[kh], E[ka] = eh + delta, ea - delta
    out = df.copy()
    out["elo_diff"] = diff
    return out, E


# ---------------------------------------------------------------- modèle
def fit_predict(train, test, ref_date, half_life=HALF_LIFE, alpha=ALPHA):
    """Entraîne sur 'train' (passé uniquement) et renvoie les buts attendus domicile / extérieur de 'test'."""
    teams = sorted(set(train.HomeTeam) | set(train.AwayTeam) | set(test.HomeTeam) | set(test.AwayTeam))
    enc = OneHotEncoder(categories=[teams], sparse_output=True).fit(np.array(teams, dtype=object).reshape(-1, 1))

    def design(df, home):
        t, o = (df.HomeTeam, df.AwayTeam) if home else (df.AwayTeam, df.HomeTeam)
        e = (df.elo_diff.values if home else -df.elo_diff.values) / 400
        extra = np.column_stack([np.full(len(df), 1.0 if home else 0.0), e])
        return sparse.hstack([enc.transform(t.to_numpy(dtype=object).reshape(-1, 1)),
                              enc.transform(o.to_numpy(dtype=object).reshape(-1, 1)),
                              sparse.csr_matrix(extra)]).tocsr()

    X = sparse.vstack([design(train, True), design(train, False)]).tocsr()
    y = np.concatenate([train.FTHG.values, train.FTAG.values]).astype(float)
    age = (ref_date - train.Date).dt.days.clip(lower=0).values
    w = np.tile(0.5 ** (age / half_life), 2)
    m = PoissonRegressor(alpha=alpha, max_iter=500).fit(X, y, sample_weight=w)
    return m.predict(design(test, True)), m.predict(design(test, False))


def probs(lh, la):
    lam = np.asarray(lh) + np.asarray(la)
    return pd.DataFrame({
        "lam": lam,
        "p_over25": 1 - poisson.cdf(2, lam),
        "p_under25": poisson.cdf(2, lam),
        "p_over15": 1 - poisson.cdf(1, lam),
        "p_over35": 1 - poisson.cdf(3, lam),
        "p_btts": (1 - np.exp(-lh)) * (1 - np.exp(-la)),
    })


# ---------------------------------------------------------------- backtest
def backtest(df, test_start, step_days=30, min_train=150, progress=None):
    """Walk-forward : pour chaque bloc de 30 jours, entraîne sur le passé et prédit le bloc."""
    out, divs = [], list(df["Div"].unique())
    for k, d in enumerate(divs):
        g = df[df.Div == d].sort_values("Date")
        edges = pd.date_range(pd.Timestamp(test_start), g.Date.max() + pd.Timedelta(days=step_days),
                              freq=f"{step_days}D")
        for a, b in zip(edges[:-1], edges[1:]):
            test, train = g[(g.Date >= a) & (g.Date < b)], g[g.Date < a]
            if len(test) == 0 or len(train) < min_train:
                continue
            lh, la = fit_predict(train, test, a)
            pr = probs(lh, la)
            pr.index = test.index
            out.append(test.join(pr))
        if progress:
            progress((k + 1) / len(divs))
    if not out:
        raise RuntimeError("Pas assez de données pour le backtest.")
    return pd.concat(out)


def _brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def evaluate(bt, w_model=0.5, thresholds=(0.0, 0.03, 0.05, 0.08)):
    d = bt.copy()
    d["y"] = ((d.FTHG + d.FTAG) > 2).astype(int)
    d["p_mkt"] = devig(d.o_over, d.o_under)
    d["p_cl"] = devig(d.c_over, d.c_under)
    d["p_fin"] = np.where(d.p_mkt.notna(), w_model * d.p_over25 + (1 - w_model) * d.p_mkt, d.p_over25)

    s = d.dropna(subset=["p_mkt", "p_cl"])
    summary = pd.DataFrame([
        ("Toujours la moyenne (référence)", _brier(s.y, np.full(len(s), s.y.mean()))),
        ("Votre modèle seul", _brier(s.y, s.p_over25)),
        ("Marché avant-match (sans marge)", _brier(s.y, s.p_mkt)),
        ("Mix modèle + marché", _brier(s.y, s.p_fin)),
        ("Marché à la clôture (le plus dur à battre)", _brier(s.y, s.p_cl)),
    ], columns=["Prédicteur", "Score de Brier (plus bas = mieux)"])
    summary["Matchs"] = len(s)

    calib = d.assign(tranche=pd.qcut(d.p_fin, 8, duplicates="drop")).groupby("tranche", observed=True).agg(
        matchs=("y", "size"), prédit=("p_fin", "mean"), réel=("y", "mean")).reset_index()
    calib["tranche"] = calib["tranche"].astype(str)

    ov = pd.DataFrame({"p": d.p_fin, "odds": d.b_over, "win": d.y == 1, "cl": d.p_cl})
    un = pd.DataFrame({"p": 1 - d.p_fin, "odds": d.b_under, "win": d.y == 0, "cl": 1 - d.p_cl})
    b = pd.concat([ov, un]).dropna(subset=["odds", "p"])
    b["edge"] = b.p * b.odds - 1
    b["profit"] = np.where(b.win, b.odds - 1, -1.0)
    b["clv"] = b.odds * b.cl - 1
    rows = []
    for t in thresholds:
        x = b[b.edge >= t]
        rows.append({"Avantage minimum": f"{t:.0%}", "Paris": len(x),
                     "Réussite %": round(100 * x.win.mean(), 1) if len(x) else np.nan,
                     "ROI %": round(100 * x.profit.mean(), 1) if len(x) else np.nan,
                     "CLV %": round(100 * x.clv.mean(), 1) if x.clv.notna().any() else np.nan})
    return summary, calib, pd.DataFrame(rows)


# ---------------------------------------------------------------- matchs à venir
def predict_upcoming(hist, E, fx, w_model=0.5, edge_min=0.05):
    res = []
    for d, g in fx.groupby("Div"):
        train = hist[hist.Div == d]
        if len(train) < 150:
            continue
        g = g.copy()
        g["elo_diff"] = [E.get((d, h), 1500.0) - E.get((d, a), 1500.0) for h, a in zip(g.HomeTeam, g.AwayTeam)]
        lh, la = fit_predict(train, g, pd.Timestamp.today().normalize())
        pr = probs(lh, la)
        pr.index = g.index
        res.append(g.join(pr))
    if not res:
        return pd.DataFrame()
    o = pd.concat(res)
    o["p_mkt"] = devig(o.o_over, o.o_under)
    o["p_over"] = np.where(o.p_mkt.notna(), w_model * o.p_over25 + (1 - w_model) * o.p_mkt, o.p_over25)
    o["p_under"] = 1 - o.p_over
    o["edge_over"] = o.p_over * o.b_over - 1
    o["edge_under"] = o.p_under * o.b_under - 1
    for s in ("over", "under"):
        e, odds = o[f"edge_{s}"], o[f"b_{s}"]
        kelly = (e / (odds - 1) / 4).clip(0, 0.02) * 100   # quart de Kelly, plafonné à 2 % de la bankroll
        o[f"stake_{s}"] = np.where(e >= edge_min, kelly, 0.0)
    return o.sort_values(["Date", "p_over"], ascending=[True, False])


# ---------------------------------------------------------------- journal de paris
JOURNAL_COLS = ["id", "date_match", "div", "domicile", "exterieur", "pari", "cote_prise", "mise",
                "bookmaker", "p_modele", "resultat", "buts", "cote_cloture_juste", "clv_pct", "profit"]


def empty_journal():
    return pd.DataFrame(columns=JOURNAL_COLS)


def add_bet(journal, match, pari, cote, mise, bookmaker=""):
    """match : une ligne du tableau renvoyé par predict_upcoming."""
    p = match["p_over"] if pari == "Over 2.5" else match["p_under"]
    nid = int(pd.to_numeric(journal["id"]).max()) + 1 if len(journal) else 1
    new = pd.DataFrame([{
        "id": nid, "date_match": match["Date"].strftime("%Y-%m-%d"), "div": match["Div"],
        "domicile": match["HomeTeam"], "exterieur": match["AwayTeam"], "pari": pari,
        "cote_prise": float(cote), "mise": float(mise), "bookmaker": bookmaker,
        "p_modele": round(float(p), 4), "resultat": "En attente", "buts": np.nan,
        "cote_cloture_juste": np.nan, "clv_pct": np.nan, "profit": np.nan}])
    return new if journal.empty else pd.concat([journal, new], ignore_index=True)[JOURNAL_COLS]


def update_journal(journal, hist):
    """Récupère score, résultat, cote juste de clôture (Pinnacle) et CLV des paris terminés."""
    j = journal.copy()
    h = hist.copy()
    h["_d"] = h.Date.dt.normalize()
    done = 0
    for i, r in j.iterrows():
        if r["resultat"] != "En attente":
            continue
        d = pd.Timestamp(r["date_match"])
        m = h[(h.Div == r["div"]) & (h.HomeTeam == r["domicile"]) & (h.AwayTeam == r["exterieur"])
              & ((h["_d"] - d).abs() <= pd.Timedelta(days=1))]
        if m.empty:
            continue
        m = m.iloc[0]
        total = int(m.FTHG + m.FTAG)
        over = r["pari"] == "Over 2.5"
        win = total > 2 if over else total <= 2
        cote, mise = float(r["cote_prise"]), float(r["mise"])
        j.at[i, "resultat"] = "Gagné" if win else "Perdu"
        j.at[i, "buts"] = total
        j.at[i, "profit"] = round(mise * (cote - 1) if win else -mise, 2)
        if pd.notna(m.c_over) and pd.notna(m.c_under):
            pc = float(devig(m.c_over, m.c_under))
            ps = pc if over else 1 - pc
            j.at[i, "cote_cloture_juste"] = round(1 / ps, 3)
            j.at[i, "clv_pct"] = round((cote * ps - 1) * 100, 2)
        done += 1
    return j, done


def journal_stats(journal):
    j = journal.copy()
    for c in ["cote_prise", "mise", "profit", "clv_pct"]:
        j[c] = pd.to_numeric(j[c], errors="coerce")
    s = j[j["resultat"].isin(["Gagné", "Perdu"])]
    c = s["clv_pct"].dropna()
    staked = s["mise"].sum()
    return {
        "paris": len(j), "termines": len(s), "en_attente": int((j["resultat"] == "En attente").sum()),
        "profit": float(s["profit"].sum()),
        "roi": float(s["profit"].sum() / staked * 100) if staked > 0 else np.nan,
        "reussite": float((s["resultat"] == "Gagné").mean() * 100) if len(s) else np.nan,
        "n_clv": len(c), "clv": float(c.mean()) if len(c) else np.nan,
        "clv_ic": float(1.96 * c.std(ddof=1) / np.sqrt(len(c))) if len(c) > 1 else np.nan,
        "serie": s.assign(date_match=s["date_match"].astype(str)).sort_values("date_match"),
    }
