import streamlit as st
import pandas as pd
import re
import json
import time
from google_play_scraper import search as gplay_search, reviews as gplay_reviews, Sort
from google import genai
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans

st.set_page_config(page_title="SignalLens", page_icon="🔍", layout="wide")

CATEGORIES = [
    "delivery_problem", "refund_payment_issue", "support_experience",
    "app_bug_technical", "pricing_complaint", "delivery_partner_behavior",
    "feature_request", "positive_experience", "noise_unclear"
]
MODEL_FALLBACK_LIST = ["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.1-flash-lite"]
MIN_REVIEWS_REQUIRED = 50
TARGET_REVIEW_COUNT = 150

CLASSIFY_PROMPT = """Classify each of these {n} app reviews into exactly ONE category from this list:

- delivery_problem: late, wrong, missing, or damaged delivery/order — use when no more specific cause is identified
- refund_payment_issue: use ONLY when refund/money/payment is the CENTRAL, explicit subject — NOT when refund is just a trailing detail after a more specific named cause
- support_experience: complaint is about the support INTERACTION itself (no human agent, bot loop, agent rude) — NOT when a specific root problem is named elsewhere
- app_bug_technical: app crashes, wrong ETA, tracking issues, technical glitches
- pricing_complaint: fees, surge pricing, discounts, cancellation charges, cost complaints
- delivery_partner_behavior: complaint specifically names a delivery/service person's conduct — takes priority over a trailing refund mention
- feature_request: suggestion to add or improve a feature
- positive_experience: praise, satisfaction
- noise_unclear: too vague/short to classify

Decision approach: identify the MOST SPECIFIC named cause, not just any word that appears. A trailing "no refund" after a specific cause is already named does NOT override that cause.

Reviews:
{numbered}

Respond with ONLY a JSON array of {n} category strings, in order, nothing else."""

THEME_PROMPT = """You are analyzing a cluster of {n} customer reviews about "{company}" that a keyword algorithm grouped together because they're similar. Here are the top distinguishing words the algorithm found: {top_terms}

Sample reviews from this cluster:
{samples}

Based ONLY on what's actually in these reviews (do not invent facts not present), respond with ONLY a JSON object with these exact fields:
{{
  "theme_name": "a short, specific 5-10 word name for this problem",
  "observed": "1-2 sentences stating only what the reviews literally show, referencing the review count",
  "inferred": "1-2 sentences of reasonable interpretation, clearly distinguished from fact",
  "unknown": "1-2 sentences on what remains genuinely unclear or unverified from this data alone",
  "next_step": "one specific, concrete validation action a PM should take next — NEVER a build/fix instruction, always an investigation step"
}}

Respond with ONLY the JSON object, nothing else."""


def get_client():
    if "GEMINI_API_KEY" not in st.secrets:
        st.error("No Gemini API key configured. Add GEMINI_API_KEY in Streamlit Cloud's Secrets settings.")
        st.stop()
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


def call_gemini(client, prompt, max_retries_per_model=2):
    for model_name in MODEL_FALLBACK_LIST:
        for attempt in range(max_retries_per_model):
            try:
                response = client.models.generate_content(model=model_name, contents=prompt)
                text = re.sub(r'^```json\s*|\s*```$', '', response.text.strip())
                return text
            except Exception as e:
                error_str = str(e)
                if any(x in error_str for x in ["404", "NOT_FOUND", "429", "RESOURCE_EXHAUSTED"]):
                    break
                else:
                    time.sleep((attempt + 1) * 3)
    return None


def find_app(company_name):
    try:
        results = gplay_search(company_name, lang="en", country="in", n_hits=5)
        if not results:
            return None
        return results[0]  # best match
    except Exception:
        return None


def fetch_reviews(app_id, target_count=TARGET_REVIEW_COUNT):
    all_reviews = []
    token = None
    last_error = None
    for _ in range(max(1, target_count // 100)):
        try:
            batch, token = gplay_reviews(app_id, lang="en", country="in", sort=Sort.NEWEST, count=100, continuation_token=token)
        except Exception as e:
            last_error = str(e)
            break
        all_reviews.extend(batch)
        if token is None or len(all_reviews) >= target_count:
            break
    df = pd.DataFrame(all_reviews)
    if len(df) == 0:
        return df, last_error
    df = df[["reviewId", "content", "score", "at"]]
    df = df[df["content"].str.len() >= 15].reset_index(drop=True)
    df = df.drop_duplicates(subset="content").reset_index(drop=True)
    return df, None


def classify_reviews(client, df, progress_bar):
    predictions = []
    batch_size = 15
    n_batches = (len(df) + batch_size - 1) // batch_size
    for i, start in enumerate(range(0, len(df), batch_size)):
        chunk = df.iloc[start:start + batch_size]
        numbered = "\n".join([f"{j+1}. \"{r}\"" for j, r in enumerate(chunk["content"].tolist())])
        prompt = CLASSIFY_PROMPT.format(n=len(chunk), numbered=numbered)
        result_text = call_gemini(client, prompt)
        if result_text:
            try:
                results = json.loads(result_text)
                cleaned = []
                for r in results:
                    c = re.sub(r'[^a-z_]', '', r.strip().lower())
                    cleaned.append(c if c in CATEGORIES else "noise_unclear")
                if len(cleaned) == len(chunk):
                    predictions.extend(cleaned)
                else:
                    predictions.extend(["noise_unclear"] * len(chunk))
            except Exception:
                predictions.extend(["noise_unclear"] * len(chunk))
        else:
            predictions.extend(["noise_unclear"] * len(chunk))
        progress_bar.progress((i + 1) / n_batches, text=f"Classifying reviews... batch {i+1}/{n_batches}")
    df = df.copy()
    df["category"] = predictions
    return df


CUSTOM_STOPWORDS = list(ENGLISH_STOP_WORDS) + [
    'app', 'order', 'ordered', 'food', 'service', 'experience', 'customer',
    'worst', 'bad', 'pathetic', 'terrible', 'horrible', 'good', 'nice', 'best',
    'just', 'like', 'don', 'didn', 'time', 'really', 'got', 'use', 'using', 'used',
]


def cluster_category(df, category, max_clusters=4):
    subset = df[df["category"] == category].reset_index(drop=True)
    if len(subset) < 6:
        subset["sub_cluster"] = 0
        return subset, 1
    k = min(max_clusters, max(1, len(subset) // 6))
    try:
        vectorizer = TfidfVectorizer(max_features=300, stop_words=CUSTOM_STOPWORDS, ngram_range=(1, 2), min_df=1)
        X = vectorizer.fit_transform(subset["content"])
        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        subset["sub_cluster"] = km.fit_predict(X)
        terms = vectorizer.get_feature_names_out()
        cluster_terms = {}
        for cid in range(k):
            center = km.cluster_centers_[cid]
            top_idx = center.argsort()[-8:][::-1]
            cluster_terms[cid] = [terms[i] for i in top_idx]
        return subset, cluster_terms
    except ValueError:
        subset["sub_cluster"] = 0
        return subset, {0: []}


def generate_theme_analysis(client, company, subset, cluster_id, top_terms):
    cluster_df = subset[subset["sub_cluster"] == cluster_id]
    samples = "\n".join([f"- {c[:150]}" for c in cluster_df["content"].head(5)])
    prompt = THEME_PROMPT.format(
        n=len(cluster_df), company=company, top_terms=", ".join(top_terms),
        samples=samples
    )
    result_text = call_gemini(client, prompt)
    if result_text:
        try:
            return json.loads(result_text)
        except Exception:
            pass
    return {
        "theme_name": f"Cluster ({', '.join(top_terms[:3])})",
        "observed": f"{len(cluster_df)} reviews share similar language but a theme summary could not be generated.",
        "inferred": "Unable to generate — inspect raw reviews below directly.",
        "unknown": "Full analysis unavailable for this cluster.",
        "next_step": "Manually review the underlying signals below."
    }


def evidence_strength(signal_volume, max_volume, avg_rating, pct_churn):
    volume_score = min(40, (signal_volume / max_volume) * 40)
    severity_score = ((5 - avg_rating) / 4) * 30
    churn_score = pct_churn * 20
    source_score = 5
    return round(volume_score + severity_score + churn_score + source_score, 1)


def confidence_label(score):
    if score >= 55:
        return "Medium"
    elif score >= 30:
        return "Medium-Low"
    return "Low"


CHURN_PATTERN = r'(competitor|uninstall|switching|never (again|use)|deleting|delete (the |this )?app|moving away|switched to)'


def run_full_analysis(company, client):
    status = st.status("Analyzing " + company + "...", expanded=True)

    status.write("🔍 Searching Play Store...")
    app = find_app(company)
    if app is None:
        status.update(label="Not found", state="error")
        st.error(f"Couldn't find a Play Store app matching \"{company}\". Try a more exact name.")
        return None
    status.write(f"Found: **{app['title']}** ({app['appId']})")

    status.write("📥 Fetching reviews...")
    reviews_df, fetch_error = fetch_reviews(app["appId"])
    if len(reviews_df) < MIN_REVIEWS_REQUIRED:
        status.update(label="Not enough data", state="error")
        if fetch_error:
            st.error(f"Review fetch failed with an error: {fetch_error}")
        st.warning(
            f"Only found {len(reviews_df)} usable reviews for {app['title']} — below our minimum threshold of "
            f"{MIN_REVIEWS_REQUIRED}. Per SignalLens's own evidence-quality rule, we don't generate a report on "
            f"too little signal rather than fake confidence on a small sample."
        )
        return None
    status.write(f"Collected {len(reviews_df)} reviews (after removing duplicates/near-empty).")

    status.write("🏷️ Classifying reviews (this takes a few minutes)...")
    progress = st.progress(0, text="Starting classification...")
    reviews_df = classify_reviews(client, reviews_df, progress)
    progress.empty()

    status.write("🧩 Clustering into themes...")
    problem_categories = [c for c in CATEGORIES if c not in ("positive_experience", "noise_unclear", "feature_request")]
    all_opportunities = []
    all_subsets = []

    theme_progress = st.progress(0, text="Generating theme analysis...")
    total_clusters_est = sum(1 for c in problem_categories if (reviews_df["category"] == c).sum() >= 3)
    done = 0

    for category in problem_categories:
        cat_count = (reviews_df["category"] == category).sum()
        if cat_count < 3:
            continue
        subset, cluster_terms = cluster_category(reviews_df, category)
        if isinstance(cluster_terms, int):
            cluster_terms = {0: []}
        max_vol_in_cat = subset["sub_cluster"].value_counts().max()

        for cid in subset["sub_cluster"].unique():
            cluster_rows = subset[subset["sub_cluster"] == cid]
            if len(cluster_rows) < 3:
                continue
            terms = cluster_terms.get(cid, [])
            analysis = generate_theme_analysis(client, company, subset, cid, terms)

            churn_pct = cluster_rows["content"].str.lower().str.contains(CHURN_PATTERN, regex=True, na=False).mean()
            avg_rating = cluster_rows["score"].mean()
            vol = len(cluster_rows)

            all_opportunities.append({
                "theme": analysis.get("theme_name", f"{category} cluster {cid}"),
                "category": category,
                "signal_volume": vol,
                "avg_rating": avg_rating,
                "pct_severe": (cluster_rows["score"] <= 2).mean(),
                "observed": analysis.get("observed", ""),
                "inferred": analysis.get("inferred", ""),
                "unknown": analysis.get("unknown", ""),
                "next_step": analysis.get("next_step", ""),
            })
            cluster_rows = cluster_rows.copy()
            cluster_rows["theme"] = analysis.get("theme_name", f"{category} cluster {cid}")
            all_subsets.append(cluster_rows)

            done += 1
            theme_progress.progress(min(1.0, done / max(1, total_clusters_est)), text=f"Analyzed {done} themes...")

    theme_progress.empty()

    opp_df = pd.DataFrame(all_opportunities)
    if len(opp_df) == 0:
        status.update(label="No clear themes found", state="error")
        st.warning("Not enough distinct signal to form themes — try a company with more review volume.")
        return None

    max_vol = opp_df["signal_volume"].max()
    opp_df["evidence_strength"] = opp_df.apply(
        lambda r: evidence_strength(r["signal_volume"], max_vol, r["avg_rating"], 0), axis=1
    )
    opp_df["confidence"] = opp_df["evidence_strength"].apply(confidence_label)
    opp_df = opp_df.sort_values("evidence_strength", ascending=False).reset_index(drop=True)

    signals_df = pd.concat(all_subsets, ignore_index=True) if all_subsets else pd.DataFrame()

    status.update(label="Analysis complete!", state="complete")
    return {"company": app["title"], "opportunities": opp_df, "signals": signals_df, "total_reviews": len(reviews_df)}


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
            st.caption(f"Showing up to 10 of {len(theme_signals)} underlying reviews.")
            st.dataframe(
                theme_signals[["content", "score", "at"]].head(10),
                use_container_width=True, hide_index=True,
                column_config={"content": "Review text", "score": "Rating", "at": "Date"}
            )


# ---------------- UI ----------------

st.title("🔍 SignalLens")
st.caption("Evidence-backed product intelligence for lean product teams — now analyzing any company live")

st.markdown("---")

col1, col2 = st.columns([2, 1])
with col1:
    company = st.text_input("Company to analyze", placeholder="e.g. Zomato, Razorpay, Postman, Freshworks")
with col2:
    st.write("")
    st.write("")
    analyze = st.button("🔎 Analyze Product Landscape", type="primary", use_container_width=True)

st.caption(
    f"⏱️ Live analysis fetches up to {TARGET_REVIEW_COUNT} recent Play Store reviews and runs them through "
    f"classification + clustering — typically takes 2-5 minutes. Companies need at least {MIN_REVIEWS_REQUIRED} "
    f"public reviews for a meaningful report."
)

if analyze:
    if not company.strip():
        st.warning("Enter a company name to analyze.")
    else:
        client = get_client()
        result = run_full_analysis(company.strip(), client)
        if result:
            st.markdown(f"## {result['company']} — Product Intelligence Report")
            o1, o2, o3 = st.columns(3)
            o1.metric("Signals analyzed", result["total_reviews"])
            o2.metric("Opportunities found", len(result["opportunities"]))
            o3.metric("Source", "Play Store")

            with st.expander("⚠️ Data limitations — read before using this report", expanded=True):
                st.markdown("""
- **Single source** — Google Play Store only, no App Store/G2/Reddit yet.
- **Recent reviews only** — this snapshot reflects a recent window, not long-term trend data.
- **Themes and analysis text are AI-generated live** for this specific run, grounded in the actual review quotes shown in each drill-down — but not independently human-validated the way our original Swiggy report was.
                """)

            st.markdown("### Top Opportunities")
            top_n = result["opportunities"].head(8)
            for i, row in top_n.iterrows():
                render_opportunity(row, result["signals"], i + 1)
else:
    st.markdown("""
    ### How this works
    1. Enter **any company** with a Play Store presence
    2. Click **Analyze**
    3. SignalLens fetches its reviews, classifies them, clusters them into named themes, scores the evidence,
       and gives you an evidence-backed Product Opportunity Report — every claim traceable back to a real review.
    """)
