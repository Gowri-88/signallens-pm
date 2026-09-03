import streamlit as st
import pandas as pd

st.set_page_config(page_title="SignalLens", page_icon="🔍", layout="wide")

DATA_DIR = "app_data"

@st.cache_data
def load_opportunities():
    return pd.read_csv(f"{DATA_DIR}/swiggy_opportunities.csv")

@st.cache_data
def load_signals():
    return pd.read_csv(f"{DATA_DIR}/swiggy_signals.csv")


def confidence_color(conf):
    return {"Medium": "🟡", "Medium-Low": "🟠", "Low": "🔴"}.get(conf, "⚪")


def render_opportunity(row, signals_df, rank):
    theme = row["theme"]
    with st.container(border=True):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"### #{rank} — {theme}")
        with col2:
            st.metric("Evidence Strength", f"{row['evidence_strength']:.0f}/100")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Signals", int(row["signal_volume"]))
        c2.metric("Avg Rating", f"{row['avg_rating']:.2f}★")
        c3.metric("Severe (≤2★)", f"{row['pct_severe']*100:.0f}%")
        c4.markdown(f"**Confidence**  \n{confidence_color(row['confidence'])} {row['confidence']}")

        with st.expander("See full evidence-backed analysis"):
            st.markdown("**✅ What we know (observed)**")
            st.write(row["observed"])
            st.markdown("**🤔 What we infer**")
            st.write(row["inferred"])
            st.markdown("**❓ What we don't know**")
            st.write(row["unknown"])
            st.markdown("**🎯 Recommended next step**")
            st.info(row["next_step"])

            st.markdown("**💬 Evidence drill-down — underlying signals**")
            theme_signals = signals_df[signals_df["theme"] == theme]
            st.caption(f"Showing up to 10 of {len(theme_signals)} underlying reviews. Every claim above traces back to real rows in this table.")
            display_cols = ["content", "score", "at", "model_used"]
            st.dataframe(
                theme_signals[display_cols].head(10),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "content": "Review text",
                    "score": "Rating",
                    "at": "Date",
                    "model_used": "Classified by",
                }
            )


# ---------------- UI ----------------

st.title("🔍 SignalLens")
st.caption("Evidence-backed product intelligence for lean product teams")

st.markdown("---")

col1, col2 = st.columns([2, 1])
with col1:
    company = st.text_input("Your company", placeholder="e.g. Swiggy")
with col2:
    st.write("")
    st.write("")
    analyze = st.button("🔎 Analyze Product Landscape", type="primary", use_container_width=True)

competitors = st.text_input("Competitors (optional, comma-separated)", placeholder="e.g. Zomato, Zepto")

if analyze:
    if not company.strip():
        st.warning("Enter a company name to analyze.")
    elif company.strip().lower() != "swiggy":
        st.error(
            f"**\"{company}\" hasn't been analyzed yet.** This is an early build of SignalLens — "
            f"right now it only has real, evidence-backed data for **Swiggy** (761 public Play Store reviews, "
            f"collected and analyzed end-to-end). \n\nTry entering **Swiggy** to see a full report, or check back "
            f"as more companies get added."
        )
    else:
        opportunities = load_opportunities()
        signals = load_signals()

        st.success("Analysis complete.")

        st.markdown("## Swiggy — Product Intelligence Report")

        o1, o2, o3, o4 = st.columns(4)
        o1.metric("Signals analyzed", len(signals))
        o2.metric("Opportunities found", len(opportunities))
        o3.metric("Date range", "5 days")
        o4.metric("Source", "Play Store")

        with st.expander("⚠️ Data limitations — read before using this report", expanded=True):
            st.markdown("""
- **Single source.** All signals come from Google Play Store only — no App Store, G2, or Reddit data yet.
- **Narrow date window (5 days).** Not enough history for trend analysis — no theme below is labeled "increasing" or "decreasing," and staleness (whether a problem was already fixed in a newer release) can't be checked yet.
- **Mixed classification provenance.** ~33% of signals were classified by a human-validated AI pipeline (84% accuracy against a labeled test set); the rest by a faster rule-based fallback, spot-checked but not independently validated to the same standard. See the `Classified by` column in each evidence drill-down.
- **Single-company mode.** No competitor data was collected in this run, so no competitive comparison is shown — better to omit it than fabricate a thin one.
            """)

        st.markdown("### Top Opportunities")
        st.caption("Ranked by evidence strength. Click into any card for the full evidence-backed breakdown.")

        top_n = opportunities.sort_values("evidence_strength", ascending=False).head(8).reset_index(drop=True)
        for i, row in top_n.iterrows():
            render_opportunity(row, signals, i + 1)

        if competitors.strip():
            st.info(
                f"You entered competitors ({competitors}) — competitor data collection isn't wired up yet in this "
                f"build. This report is running in single-company mode."
            )

else:
    st.markdown("""
    ### How this works
    1. Enter a company name (try **Swiggy** — it's the only one with real data right now)
    2. Optionally list competitors
    3. Click **Analyze** to get an evidence-backed report: top customer problems, how strong the evidence is,
       and what's worth investigating next — with every claim traceable back to a real review.
    """)
