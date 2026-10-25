import numpy as np
import pandas as pd
import streamlit as st

import model as M

st.set_page_config(page_title="Over/Under 2.5", page_icon="⚽", layout="centered")
st.title("⚽ Over/Under 2.5 buts")
st.caption("Poisson régularisé + Élo + marché. Commencez par l’onglet Backtest pour savoir si le modèle vaut quelque chose.")

with st.sidebar:
    st.header("Réglages")
    divs = st.multiselect("Championnats", list(M.LEAGUES), default=["E0", "SP1", "I1", "D1", "F1"],
                          format_func=lambda d: M.LEAGUES[d])
    n_seasons = st.slider("Saisons de données", 3, 7, 5)
    auto = st.checkbox("Calibrage et poids automatiques (recommandé)", True)
    w_manual = st.slider("Poids du modèle (le reste = marché)", 0.0, 1.0, 0.5, 0.05, disabled=auto)
    edge_min = st.slider("Avantage minimum pour jouer", 0.0, 0.15, 0.05, 0.01, format="%.2f")
    days = st.slider("Matchs des X prochains jours", 1, 14, 3)

if not divs:
    st.warning("Choisissez au moins un championnat dans le menu à gauche (flèche en haut).")
    st.stop()


@st.cache_resource
def _last_good():
    return {}


def safe(fn, key, *args):
    """Exécute fn ; si le site est injoignable, renvoie la dernière lecture réussie (stale=True)."""
    store = _last_good()
    try:
        res = fn(*args)
        store[key] = res
        return res, False
    except Exception:
        if key in store:
            return store[key], True
        raise


@st.cache_data(ttl=6 * 3600, show_spinner="Téléchargement des données…")
def load(divs_t, n):
    failed = []
    df = M.load_results(list(divs_t), n, failed=failed)
    hist, elo = M.add_elo(df)
    return hist, elo, failed


@st.cache_data(ttl=3600, show_spinner="Matchs à venir…")
def fixtures(divs_t):
    report = {}
    fx_ = M.load_fixtures(list(divs_t), report=report)
    return fx_, report


CACHE_V = 3  # à augmenter si le calcul du calibrage change : force un nouveau calcul au lieu de réutiliser l'ancien


@st.cache_data(ttl=24 * 3600, show_spinner="Calibrage du modèle sur le passé (environ 1 minute, une fois par jour)…")
def tune(divs_t, n, v=CACHE_V):
    hist_, _, _ = load(divs_t, n)
    y0 = int(hist_.Season.max())
    return M.auto_tune(M.backtest(hist_, f"{y0 - 1}-08-01"))


try:
    (hist, elo, failed), stale = safe(load, ("load", tuple(divs), n_seasons), tuple(divs), n_seasons)
except Exception as e:
    st.error(f"Impossible de charger les données : {e}")
    st.stop()
if stale:
    st.warning("Le site de données est injoignable : la dernière version enregistrée est utilisée.")
if failed:
    st.caption(f"{len(failed)} fichier(s) de données indisponible(s) (par exemple : {', '.join(failed[:4])}).")

tuned = None
if auto:
    try:
        tuned = tune(tuple(divs), n_seasons)
        if "btts" not in tuned:  # résultat d'une ancienne version resté en mémoire : on recalcule
            tune.clear()
            tuned = tune(tuple(divs), n_seasons)
    except Exception:
        st.warning("Calibrage automatique indisponible pour le moment : réglages par défaut utilisés.")
cal = tuned["cal"] if tuned else M.IDENTITY
w_model = tuned["w"] if tuned else w_manual
cal_btts = tuned["cal_btts"] if tuned else M.IDENTITY
st.sidebar.caption(f"Poids du modèle appliqué : {w_model:.0%} · Calibrage : "
                   f"{'oui' if tuned and tuned['cal_applied'] else 'non'}")

up = pd.DataFrame()
diag = {}
try:
    (fx_all, rep), fstale = safe(fixtures, ("fx", tuple(divs)), tuple(divs))
    today = pd.Timestamp.today().normalize()
    fx = fx_all[(fx_all.Date >= today) & (fx_all.Date <= today + pd.Timedelta(days=days))]
    diag = {"rep": rep, "fx_all": fx_all, "n_window": len(fx), "today": today}
    if fstale:
        st.caption("Matchs à venir : dernière version enregistrée (site injoignable).")
    if not fx.empty:
        with st.spinner("Calcul des probabilités…"):
            up = M.predict_upcoming(hist, elo, fx, w_model, edge_min, cal, cal_btts)
except Exception as e:
    st.warning("Les matchs à venir ne sont pas disponibles pour le moment. Réessayez dans quelques minutes.")
    st.caption(f"Détail technique : {e}")

tab1, tab_j, tab2, tab3 = st.tabs(["Prédictions", "Journal", "Backtest", "À propos"])

# ------------------------------------------------------------- prédictions
with tab1:
    if up.empty:
        st.info("Aucun match à venir sur cette période pour ces championnats.")
        if diag:
            with st.expander("Diagnostic : pourquoi aucun match ?", expanded=True):
                rep, fx_all, today = diag["rep"], diag["fx_all"], diag["today"]
                if rep:
                    d1 = rep["dmin"].strftime("%d/%m/%Y") if rep.get("dmin") is not None and pd.notna(rep.get("dmin")) else "?"
                    d2 = rep["dmax"].strftime("%d/%m/%Y") if rep.get("dmax") is not None and pd.notna(rep.get("dmax")) else "?"
                    st.write(f"Fichier du site : **{rep['rows']} matchs**, du {d1} au {d2}.")
                    st.write("Championnats présents : " + (", ".join(f"{M.LEAGUES.get(k, k)} ({v})" for k, v in rep["divs"].items()) or "aucun"))
                st.write(f"Matchs de vos championnats dans ce fichier : **{len(fx_all)}** ; dans les {days} prochains jours : **{diag['n_window']}**.")
                if rep and rep["rows"] == 0:
                    st.error("Cause : le fichier du site est vide. Le calendrier n’est pas publié pour le moment.")
                elif rep and rep.get("dmax") is not None and pd.notna(rep["dmax"]) and rep["dmax"] < today:
                    st.error("Cause : le fichier du site est périmé (tous ses matchs sont déjà passés). Il sera mis à jour par le site.")
                elif len(fx_all) == 0:
                    st.error("Cause : aucun de vos championnats ne figure dans le fichier. Le site ne couvre pas tous les championnats à venir.")
                elif diag["n_window"] == 0:
                    nxt = fx_all[fx_all.Date >= today].Date.min()
                    st.warning("Cause : le site ne publie que les prochaines journées. " +
                               (f"Prochain match connu : {nxt:%d/%m/%Y}." if pd.notna(nxt) else "Aucun match futur dans le fichier."))
                else:
                    thin = [M.LEAGUES.get(d, d) for d in fx_all.Div.unique() if (hist.Div == d).sum() < 150]
                    st.warning("Cause probable : historique insuffisant pour " + (", ".join(thin) if thin else "certains championnats")
                               + ". Augmentez « Saisons de données » dans le menu de gauche.")
    else:
        c1, c2 = st.columns(2)
        only_value = c1.checkbox("Seulement les paris avec value", False)
        with_btts = c2.checkbox("Afficher aussi BTTS (sans cote bookmaker)", False)
        if with_btts and tuned:
            rb = tuned["btts"]
            fin = rb["b_cal"] if rb["applied"] else rb["b_raw"]
            skill = rb["b_base"] - fin
            if skill <= 0:
                st.error("BTTS : sur des matchs jamais vus, le modèle ne fait pas mieux que la simple moyenne. "
                         "Ne vous fiez pas à ces probabilités.")
            elif skill < 0.002:
                st.warning("BTTS : signal très faible sur des matchs jamais vus. À utiliser avec prudence.")
            else:
                st.info("BTTS : le modèle fait mieux que la moyenne, mais aucune cote BTTS n’est disponible "
                        "dans les données : impossible de mesurer une value.")
        rows = []
        for r in up.itertuples():
            match = f"{r.Date:%d/%m} · {r.HomeTeam} – {r.AwayTeam}"
            bets = [("Over 2.5", r.p_over, r.b_over, r.edge_over), ("Under 2.5", r.p_under, r.b_under, r.edge_under)]
            if with_btts:
                bets += [("BTTS Oui", r.p_btts, np.nan, np.nan), ("BTTS Non", 1 - r.p_btts, np.nan, np.nan)]
            for name, p, odds, edge in bets:
                has_value = pd.notna(edge) and edge > 0 and edge >= edge_min
                if only_value and not has_value:
                    continue
                rows.append({"Match": match, "Type de pari": f"{name} · {p * 100:.0f} %",
                             "Cote juste": 1 / p, "Cote bookmaker": odds,
                             "Expected Value": f"{edge * 100:+.1f} %" if has_value else ""})
        if not rows:
            st.info("Aucun pari avec value pour le moment. C’est le cas le plus fréquent.")
        else:
            show = pd.DataFrame(rows)
            styled = (show.style
                      .format({"Cote juste": "{:.2f}", "Cote bookmaker": "{:.2f}"}, na_rep="–")
                      .apply(lambda r: ["background-color: rgba(46,160,67,0.22)" if r["Expected Value"] else ""] * len(r),
                             axis=1))
            st.dataframe(styled, hide_index=True, width="stretch")
        st.caption("Cote juste = 1 ÷ probabilité (mix modèle + marché, réglé automatiquement). "
                   "Cote bookmaker = moyenne du marché. Expected Value = gain moyen attendu par euro misé, "
                   "affiché seulement si l’avantage dépasse le seuil du menu de gauche (ligne en vert).")
        with st.expander("➕ Enregistrer un pari dans mon journal"):
            rows = list(up.itertuples())
            labels = [f"{r.Date:%d/%m} – {r.HomeTeam} – {r.AwayTeam}" for r in rows]
            i = st.selectbox("Match", range(len(rows)), format_func=lambda k: labels[k])
            pari = st.radio("Pari", ["Over 2.5", "Under 2.5"], horizontal=True)
            row = up.iloc[i]
            dflt = row["b_over"] if pari == "Over 2.5" else row["b_under"]
            cote = st.number_input("Cote réellement obtenue", 1.01, 20.0,
                                   float(dflt) if pd.notna(dflt) else 1.90, 0.01)
            mise = st.number_input("Mise (€)", 0.5, 10000.0, 10.0, 0.5)
            book = st.text_input("Bookmaker (facultatif)")
            if st.button("Ajouter au journal"):
                jr = st.session_state.get("journal", M.empty_journal())
                st.session_state["journal"] = M.add_bet(jr, row, pari, cote, mise, book)
                st.success("Pari ajouté. Pensez à télécharger votre journal dans l’onglet Journal.")

# ------------------------------------------------------------- journal
with tab_j:
    st.write("Le journal enregistre chaque pari réel et calcule votre **CLV** : votre cote comparée à la cote "
             "juste de clôture de Pinnacle. C’est le test le plus fiable pour savoir si vous avez un avantage.")
    st.warning("Le journal n’est conservé que pendant votre visite. **Téléchargez-le après chaque ajout** "
               "et réimportez-le à la visite suivante.")
    if "journal" not in st.session_state:
        st.session_state["journal"] = M.empty_journal()
    f = st.file_uploader("Importer mon journal (fichier CSV)", type="csv")
    if f is not None and st.session_state.get("journal_sig") != (f.name, f.size):
        try:
            imp = pd.read_csv(f)
            st.session_state["journal"] = imp.reindex(columns=M.JOURNAL_COLS)
            st.session_state["journal_sig"] = (f.name, f.size)
            st.success(f"{len(imp)} pari(s) importé(s).")
        except Exception as e:
            st.error(f"Fichier illisible : {e}")
    j = st.session_state["journal"]
    c1, c2 = st.columns(2)
    if c1.button("Mettre à jour les résultats"):
        j, n = M.update_journal(j, hist)
        st.session_state["journal"] = j
        st.success(f"{n} pari(s) terminé(s) mis à jour." if n else
                   "Aucun nouveau résultat (les données du site arrivent environ 1 jour après le match).")
    c2.download_button("Télécharger mon journal", j.to_csv(index=False).encode("utf-8"),
                       "journal_paris.csv", "text/csv")
    if j.empty:
        st.info("Journal vide. Ajoutez un pari depuis l’onglet Prédictions.")
    else:
        st_ = M.journal_stats(j)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Paris terminés", f"{st_['termines']} / {st_['paris']}")
        k2.metric("Profit", f"{st_['profit']:+.2f} €")
        k3.metric("ROI", "–" if pd.isna(st_["roi"]) else f"{st_['roi']:+.1f} %")
        k4.metric("CLV moyen", "–" if pd.isna(st_["clv"]) else f"{st_['clv']:+.1f} %",
                  None if pd.isna(st_["clv_ic"]) else f"± {st_['clv_ic']:.1f}", delta_color="off")
        if st_["n_clv"] < 100:
            st.info(f"Trop tôt pour conclure : {st_['n_clv']} paris avec CLV. Visez 100 au minimum, idéalement 300 ou plus.")
        elif st_["clv"] - st_["clv_ic"] > 0:
            st.success("CLV positif et statistiquement distinct de zéro : signe d’un vrai avantage face au marché.")
        elif st_["clv"] > 0:
            st.warning("CLV positif mais pas encore distinguable du hasard. Continuez à accumuler des paris.")
        else:
            st.error("CLV négatif ou nul : pas d’avantage démontré. Ne misez pas davantage.")
        ser = st_["serie"]
        if len(ser) > 1:
            st.line_chart(ser.assign(Profit_cumulé=ser["profit"].cumsum()).set_index("date_match")["Profit_cumulé"])
        st.caption("Vous pouvez corriger la cote, la mise ou le bookmaker dans le tableau, ou supprimer une ligne.")
        locked = [c for c in M.JOURNAL_COLS if c not in ("cote_prise", "mise", "bookmaker")]
        st.session_state["journal"] = st.data_editor(j, num_rows="dynamic", hide_index=True,
                                                     disabled=locked, key="journal_editor", width="stretch")

# ------------------------------------------------------------- backtest
with tab2:
    st.write("Le backtest rejoue le passé : pour chaque mois, le modèle n’utilise que les matchs déjà joués, "
             "puis on compare ses prévisions aux résultats réels et aux cotes du marché.")
    if tuned:
        with st.expander("Réglages automatiques en vigueur", expanded=True):
            st.write(f"**Poids du modèle retenu : {tuned['w']:.0%}** (le reste = marché).")
            if tuned["cal_applied"]:
                st.write(f"**Calibrage appliqué** : l’erreur passe de {tuned['brier_raw_holdout']:.4f} à "
                         f"{tuned['brier_cal_holdout']:.4f} sur des matchs non utilisés pour le réglage.")
            else:
                st.write("**Calibrage non appliqué** : il n’améliore pas les matchs jamais vus.")
            st.caption("Réglage fait sur la saison passée et la saison en cours. Le tableau ci-dessous est mesuré "
                       "sur ces mêmes matchs : il est donc légèrement optimiste.")
    if tuned:
        rb = tuned["btts"]
        with st.expander("BTTS : fiabilité mesurée", expanded=True):
            fin = rb["b_cal"] if rb["applied"] else rb["b_raw"]
            st.write(f"Erreur (Brier) sur des matchs non utilisés pour le réglage : modèle **{fin:.4f}** contre "
                     f"**{rb['b_base']:.4f}** pour « toujours la moyenne » "
                     f"({'calibrage appliqué' if rb['applied'] else 'sans calibrage'}).")
            if rb["b_base"] - fin <= 0:
                st.error("Le modèle ne fait pas mieux que la moyenne sur le BTTS : ne vous fiez pas à ses probabilités.")
            elif rb["b_base"] - fin < 0.002:
                st.warning("Signal très faible : à utiliser avec prudence.")
            else:
                st.success("Le modèle fait mieux que la moyenne. Les données ne contiennent pas de cotes BTTS : "
                           "on ne peut pas savoir s’il bat le marché.")
    k = st.radio("Période testée", [1, 2, 3], index=1, horizontal=True,
                 format_func=lambda x: f"{x} dernière{'s' if x > 1 else ''} saison{'s' if x > 1 else ''}")
    if st.button("Lancer le backtest", type="primary"):
        y0 = int(hist.Season.max())
        bar = st.progress(0.0)
        try:
            st.session_state["bt"] = M.backtest(hist, f"{y0 - (k - 1)}-08-01", progress=bar.progress)
        except Exception as e:
            st.error(str(e))
        bar.empty()
    if "bt" in st.session_state:
        summary, calib, bets = M.evaluate(st.session_state["bt"], w_model, cal=cal)
        st.subheader("Qui prédit le mieux ?")
        st.dataframe(summary, hide_index=True, width="stretch")
        m = summary.set_index("Prédicteur")["Score de Brier (plus bas = mieux)"]
        mdl, mkt = m["Votre modèle recalibré"], m["Marché avant-match (sans marge)"]
        if mdl >= mkt:
            st.warning("Le modèle seul prédit moins bien que le marché : c’est le cas habituel. "
                       "Gardez un poids du marché élevé et ne misez pas sur le modèle seul.")
        else:
            st.success("Le modèle seul bat le marché avant-match sur cette période : résultat rare, à confirmer.")
        st.subheader("Calibration : les % annoncés sont-ils justes ?")
        st.dataframe(calib, hide_index=True, width="stretch")
        st.subheader("Simulation de paris (mise fixe, cotes moyennes)")
        st.dataframe(bets, hide_index=True, width="stretch")
        st.subheader("Par championnat")
        st.dataframe(M.by_league(st.session_state["bt"], w_model, cal, cal_btts, edge_min),
                     hide_index=True, width="stretch")
        st.caption("Écart négatif = le modèle fait mieux. Un écart n’est retenu que s’il dépasse sa marge d’erreur "
                   "(2 écarts-types), sinon « indiscernable ». Avec peu de paris value, le CLV par championnat reste très bruité.")
        st.caption("CLV = votre cote comparée à la cote juste de clôture. C’est le meilleur indicateur d’un vrai "
                   "avantage : un CLV positif et stable sur plusieurs centaines de paris compte plus qu’un ROI "
                   "flatteur obtenu par chance.")

# ------------------------------------------------------------- à propos
with tab3:
    st.markdown(
        """
**Données** : football-data.co.uk (résultats, cotes Pinnacle / moyennes, matchs à venir). Aucune clé API nécessaire.

**Modèle** : régression de Poisson régularisée (force d’attaque et de défense, avantage du terrain),
matchs récents pondérés davantage, écart d’Élo calculé avant chaque match. Le total de buts suit une loi de Poisson.
La probabilité d’Over 2.5 est recalibrée automatiquement si cela améliore des matchs jamais vus, puis mélangée au marché sans marge avec un poids choisi par le backtest (le marché reste privilégié à égalité).

**Limites** : pas de xG, pas de blessés ni de compositions. Un avantage durable face au marché est rare ;
n’engagez que de l’argent que vous pouvez perdre.
        """
    )
