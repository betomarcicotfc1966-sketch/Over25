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
    w_model = st.slider("Poids du modèle (le reste = marché)", 0.0, 1.0, 0.5, 0.05)
    edge_min = st.slider("Avantage minimum pour jouer", 0.0, 0.15, 0.05, 0.01, format="%.2f")
    days = st.slider("Matchs des X prochains jours", 1, 14, 3)

if not divs:
    st.warning("Choisissez au moins un championnat dans le menu à gauche (flèche en haut).")
    st.stop()


@st.cache_data(ttl=6 * 3600, show_spinner="Téléchargement des données…")
def load(divs_t, n):
    df = M.load_results(list(divs_t), n)
    return M.add_elo(df)


@st.cache_data(ttl=3600, show_spinner="Matchs à venir…")
def fixtures(divs_t):
    return M.load_fixtures(list(divs_t))


try:
    hist, elo = load(tuple(divs), n_seasons)
except Exception as e:
    st.error(f"Impossible de charger les données : {e}")
    st.stop()

up = pd.DataFrame()
try:
    fx = fixtures(tuple(divs))
    today = pd.Timestamp.today().normalize()
    fx = fx[(fx.Date >= today) & (fx.Date <= today + pd.Timedelta(days=days))]
    if not fx.empty:
        with st.spinner("Calcul des probabilités…"):
            up = M.predict_upcoming(hist, elo, fx, w_model, edge_min)
except Exception as e:
    st.error(f"Matchs à venir indisponibles : {e}")

tab1, tab_j, tab2, tab3 = st.tabs(["Prédictions", "Journal", "Backtest", "À propos"])

# ------------------------------------------------------------- prédictions
with tab1:
    if up.empty:
        st.info("Aucun match à venir sur cette période pour ces championnats (ou pas assez de données).")
    else:
        def tag(r):
            if r.stake_over > 0:
                return f"✅ Over ({r.stake_over:.1f} %)"
            if r.stake_under > 0:
                return f"✅ Under ({r.stake_under:.1f} %)"
            return ""
        show = pd.DataFrame({
            "Date": up.Date.dt.strftime("%d/%m"),
            "Match": up.HomeTeam + " – " + up.AwayTeam,
            "Value": up.apply(tag, axis=1),
            "Over 2.5 %": (up.p_over * 100).round(1),
            "Cote juste O": (1 / up.p_over).round(2),
            "Under 2.5 %": (up.p_under * 100).round(1),
            "Cote juste U": (1 / up.p_under).round(2),
            "BTTS %": (up.p_btts * 100).round(1),
            "Cote juste BTTS": (1 / up.p_btts).round(2),
            "Cote O": up.b_over.round(2),
            "Avantage O %": (up.edge_over * 100).round(1),
            "Cote U": up.b_under.round(2),
            "Avantage U %": (up.edge_under * 100).round(1),
        })
        st.dataframe(show, hide_index=True, width="stretch")
        st.caption("Probabilités = mix modèle + marché (réglable). « Value » : avantage ≥ seuil, "
                   "mise suggérée en % de la bankroll (quart de Kelly, plafonnée à 2 %). "
                   "Cotes = moyenne du marché. Les probabilités BTTS viennent du modèle seul.")
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
        summary, calib, bets = M.evaluate(st.session_state["bt"], w_model)
        st.subheader("Qui prédit le mieux ?")
        st.dataframe(summary, hide_index=True, width="stretch")
        m = summary.set_index("Prédicteur")["Score de Brier (plus bas = mieux)"]
        mdl, mkt = m["Votre modèle seul"], m["Marché avant-match (sans marge)"]
        if mdl >= mkt:
            st.warning("Le modèle seul prédit moins bien que le marché : c’est le cas habituel. "
                       "Gardez un poids du marché élevé et ne misez pas sur le modèle seul.")
        else:
            st.success("Le modèle seul bat le marché avant-match sur cette période : résultat rare, à confirmer.")
        st.subheader("Calibration : les % annoncés sont-ils justes ?")
        st.dataframe(calib, hide_index=True, width="stretch")
        st.subheader("Simulation de paris (mise fixe, cotes moyennes)")
        st.dataframe(bets, hide_index=True, width="stretch")
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
Les probabilités finales mélangent le modèle et le marché sans marge.

**Limites** : pas de xG, pas de blessés ni de compositions. Un avantage durable face au marché est rare ;
n’engagez que de l’argent que vous pouvez perdre.
        """
    )
