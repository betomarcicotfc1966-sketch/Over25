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

tab1, tab2, tab3 = st.tabs(["Prédictions", "Backtest", "À propos"])

# ------------------------------------------------------------- prédictions
with tab1:
    try:
        fx = fixtures(tuple(divs))
        end = pd.Timestamp.today().normalize() + pd.Timedelta(days=days)
        fx = fx[(fx.Date >= pd.Timestamp.today().normalize()) & (fx.Date <= end)]
    except Exception as e:
        fx = pd.DataFrame()
        st.error(f"Matchs à venir indisponibles : {e}")
    if fx.empty:
        st.info("Aucun match à venir sur cette période pour ces championnats.")
    else:
        with st.spinner("Calcul des probabilités…"):
            up = M.predict_upcoming(hist, elo, fx, w_model, edge_min)
        if up.empty:
            st.info("Pas assez de données pour ces championnats.")
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
