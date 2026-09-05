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


NON_CONSUMER_APP_TERMS = [
    'delivery partner', 'partner app', 'for partners', 'driver', 'captain',
    'merchant', 'seller', 'business app', 'for business', 'agent app', 'rider'
]


def find_app_candidates(company_name, n=6):
    try:
        results = gplay_search(company_name, lang="en", country="in", n_hits=10)
        valid = [r for r in results if r.get("appId")]
        if not valid:
            return []

        query = company_name.lower()

        def is_relevant(r):
            # a result only counts as relevant if the query actually appears in the
            # title or developer name — otherwise a short, totally unrelated app
            # (e.g. "Slack" for a "freshworks" search) can sneak in via a length fluke
            title = (r.get("title") or "").lower()
            developer = (r.get("developer") or "").lower()
            return query in title or query in developer

        relevant = [r for r in valid if is_relevant(r)]
        pool = relevant if relevant else valid  # fallback only if nothing matches at all

        def score(r):
            title = (r.get("title") or "").lower()
            developer = (r.get("developer") or "").lower()
            penalty = 1000 if any(term in title for term in NON_CONSUMER_APP_TERMS) else 0
            # developer name matching the company is the strongest signal (catches
            # sub-brands like Freshdesk/Freshchat all published by "Freshworks Inc")
            if query in developer:
                relevance_bonus = -500
            elif title.startswith(query):
                relevance_bonus = -200
            elif query in title:
                relevance_bonus = -100
            else:
                relevance_bonus = 0
            length_tiebreak = len(title) * 0.1  # only matters among near-ties now
            return penalty + relevance_bonus + length_tiebreak

        pool.sort(key=score)
        return pool[:n]
    except Exception:
        return []


MAX_REVIEWS_SAFETY_CAP = 400  # bounds classification time/cost even for a wide date window

def fetch_reviews(app_id, days_window=7):
    import datetime
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days_window)
    all_reviews = []
    token = None
    last_error = None
    for _ in range(20):  # safety limit on pagination loops
        try:
            batch, token = gplay_reviews(app_id, lang="en", country="in", sort=Sort.NEWEST, count=100, continuation_token=token)
        except Exception as e:
            last_error = str(e)
            break
        if not batch:
            break
        all_reviews.extend(batch)
        oldest_in_batch = min(pd.to_datetime(r["at"]) for r in batch)
        if oldest_in_batch < cutoff:
            break
        if len(all_reviews) >= MAX_REVIEWS_SAFETY_CAP:
            break
        if token is None:
            break
    df = pd.DataFrame(all_reviews)
    if len(df) == 0:
        return df, last_error, None, None
    df = df[["reviewId", "content", "score", "at"]]
    df["at"] = pd.to_datetime(df["at"])
    df = df[df["at"] >= cutoff]  # trim any overshoot past the window
    df = df[df["content"].str.len() >= 15].reset_index(drop=True)
    df = df.drop_duplicates(subset="content").reset_index(drop=True)
    if len(df) > MAX_REVIEWS_SAFETY_CAP:
        df = df.head(MAX_REVIEWS_SAFETY_CAP)
    if len(df) == 0:
        return df, None, None, None
    date_min = df["at"].min()
    date_max = df["at"].max()
    return df, None, date_min, date_max


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


def merge_duplicate_themes(client, company, opportunities):
    """One extra call at the end: check if any themes are really describing the same
    underlying problem in different words, and merge them. Plain keyword similarity isn't
    reliable enough for this (tested — paraphrases share too few exact words), so this
    needs actual semantic judgment."""
    if len(opportunities) <= 1:
        return [[i] for i in range(len(opportunities))]

    listing = "\n".join([f"{i+1}. {o['theme']} — {o['observed'][:150]}" for i, o in enumerate(opportunities)])
    prompt = f"""Here are {len(opportunities)} customer-problem themes found in reviews for "{company}". Some may describe the SAME underlying problem in different words (e.g. "SMS verification failures" and "login code delivery issues" are the same problem — OTP/verification delivery — even though they share no exact words).

Themes:
{listing}

Group these into merged clusters based on whether they describe the same real underlying problem, not just similar wording. Respond with ONLY a JSON array of groups, where each group is a list of the 1-based indices that belong together. Every index must appear in exactly one group. A group can contain just one index if it's genuinely distinct from all others.

Example format: [[1, 3, 4], [2], [5, 6]]"""

    result_text = call_gemini(client, prompt)
    if result_text:
        try:
            groups = json.loads(result_text)
            seen = set()
            valid_groups = []
            for g in groups:
                g = [i - 1 for i in g if isinstance(i, int) and 0 <= i - 1 < len(opportunities) and (i - 1) not in seen]
                if g:
                    valid_groups.append(g)
                    seen.update(g)
            for i in range(len(opportunities)):
                if i not in seen:
                    valid_groups.append([i])
            return valid_groups
        except Exception:
            pass
    # fallback: no merging if the call fails — safer than crashing the report
    return [[i] for i in range(len(opportunities))]


def apply_theme_merge(opportunities, all_subsets, groups):
    merged_opportunities = []
    merged_subsets = []
    for group in groups:
        members = [opportunities[i] for i in group]
        if len(members) == 1:
            merged_opportunities.append(members[0])
            merged_subsets.append(all_subsets[group[0]])
            continue
        # merge: keep the largest-volume member's text (safer than generating new text),
        # sum volumes, weighted-average the numeric fields
        members_sorted = sorted(members, key=lambda m: m["signal_volume"], reverse=True)
        primary = members_sorted[0]
        total_vol = sum(m["signal_volume"] for m in members)
        weighted_rating = sum(m["avg_rating"] * m["signal_volume"] for m in members) / total_vol
        weighted_severe = sum(m["pct_severe"] * m["signal_volume"] for m in members) / total_vol
        merged = dict(primary)
        merged["signal_volume"] = total_vol
        merged["avg_rating"] = weighted_rating
        merged["pct_severe"] = weighted_severe
        merged_opportunities.append(merged)

        combined_signals = pd.concat([all_subsets[i] for i in group], ignore_index=True)
        combined_signals = combined_signals.copy()
        combined_signals["theme"] = primary["theme"]  # unify theme label for drill-down
        merged_subsets.append(combined_signals)

    return merged_opportunities, merged_subsets
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


def run_full_analysis(company, app, days_window, client):
    status = st.status("Analyzing " + company + "...", expanded=True)

    status.write(f"Using: **{app['title']}** ({app['appId']})")

    status.write("📥 Fetching reviews...")
    reviews_df, fetch_error, date_min, date_max = fetch_reviews(app["appId"], days_window=days_window)
    if len(reviews_df) < MIN_REVIEWS_REQUIRED:
        status.update(label="Not enough data", state="error")
        if fetch_error is not None:
            st.error(f"Review fetch failed with an error: {fetch_error}")
        st.warning(
            f"Only found {len(reviews_df)} usable reviews for {app['title']} within the last {days_window} days — "
            f"below our minimum threshold of {MIN_REVIEWS_REQUIRED}. Try a wider time window, or this app may "
            f"not have enough public review volume for a meaningful report."
        )
        return None
    status.write(f"Collected {len(reviews_df)} reviews (after removing duplicates/near-empty).")
    date_range_str = f"{date_min.strftime('%Y-%m-%d')} to {date_max.strftime('%Y-%m-%d')}"
    date_span_days = (date_max - date_min).days
    status.write(f"📅 Date range covered: **{date_range_str}** ({date_span_days} day{'s' if date_span_days != 1 else ''})")

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

    if len(all_opportunities) == 0:
        status.update(label="No clear themes found", state="error")
        st.warning("Not enough distinct signal to form themes — try a company with more review volume.")
        return None

    status.write(f"🔗 Checking for duplicate themes ({len(all_opportunities)} found so far)...")
    groups = merge_duplicate_themes(client, company, all_opportunities)
    all_opportunities, all_subsets = apply_theme_merge(all_opportunities, all_subsets, groups)
    n_merged = sum(1 for g in groups if len(g) > 1)
    if n_merged > 0:
        status.write(f"🔗 Merged {n_merged} group(s) of duplicate themes into single opportunities.")

    opp_df = pd.DataFrame(all_opportunities)

    max_vol = opp_df["signal_volume"].max()
    opp_df["evidence_strength"] = opp_df.apply(
        lambda r: evidence_strength(r["signal_volume"], max_vol, r["avg_rating"], 0), axis=1
    )
    opp_df["confidence"] = opp_df["evidence_strength"].apply(confidence_label)
    opp_df = opp_df.sort_values("evidence_strength", ascending=False).reset_index(drop=True)

    signals_df = pd.concat(all_subsets, ignore_index=True) if all_subsets else pd.DataFrame()

    status.update(label="Analysis complete!", state="complete")
    return {
        "company": app["title"], "opportunities": opp_df, "signals": signals_df,
        "total_reviews": len(reviews_df), "date_min": date_min, "date_max": date_max
    }


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

if "candidates" not in st.session_state:
    st.session_state.candidates = None
    st.session_state.company_query = ""

col1, col2 = st.columns([2, 1])
with col1:
    company = st.text_input("Company to analyze", placeholder="e.g. Zomato, Razorpay, Postman, Freshworks")
with col2:
    st.write("")
    st.write("")
    search_clicked = st.button("🔍 Find App", type="secondary", use_container_width=True)

time_window_label = st.selectbox(
    "Time window",
    ["Last 24 hours", "Last 7 days", "Last 30 days"],
    index=1,
    help="Higher-volume apps generate hundreds of reviews/day — a short window keeps the report current. "
         "Lower-volume apps may need a wider window just to reach enough signal."
)
DAYS_MAP = {"Last 24 hours": 1, "Last 7 days": 7, "Last 30 days": 30}
days_window = DAYS_MAP[time_window_label]

st.caption(
    f"⏱️ Live analysis fetches reviews within your chosen window (capped at {MAX_REVIEWS_SAFETY_CAP} for "
    f"speed/cost) and runs them through classification + clustering — typically takes 2-5 minutes. "
    f"Companies need at least {MIN_REVIEWS_REQUIRED} public reviews in that window for a meaningful report."
)

if search_clicked and company.strip():
    with st.spinner("Searching Play Store..."):
        candidates = find_app_candidates(company.strip(), n=8)
    st.session_state.candidates = candidates
    st.session_state.company_query = company.strip()

if st.session_state.candidates:
    if len(st.session_state.candidates) == 0:
        st.error(f"Couldn't find any Play Store app matching \"{st.session_state.company_query}\". Try a different spelling, or enter the package ID directly below.")
    else:
        st.markdown(f"**Found {len(st.session_state.candidates)} possible matches — confirm the right one:**")
        st.caption("⚠️ Nothing is pre-selected. Double-check the title and package ID match what you're looking for before analyzing.")
        options = {
            f"{c['title']}  —  {c['appId']}": c for c in st.session_state.candidates
        }
        PLACEHOLDER = "— Select the correct app —"
        choice_label = st.radio("Select the correct app", [PLACEHOLDER] + list(options.keys()), index=0)

        if choice_label == PLACEHOLDER:
            st.info("Pick an app above to continue.")
        else:
            selected_app = options[choice_label]
            analyze = st.button("✅ Analyze This App", type="primary")

            if analyze:
                client = get_client()
                result = run_full_analysis(st.session_state.company_query, selected_app, days_window, client)
                if result:
                    st.markdown(f"## {result['company']} — Product Intelligence Report")
                    date_span_days = (result["date_max"] - result["date_min"]).days
                    o1, o2, o3, o4 = st.columns(4)
                    o1.metric("Signals analyzed", result["total_reviews"])
                    o2.metric("Opportunities found", len(result["opportunities"]))
                    o3.metric("Date range", f"{date_span_days} day{'s' if date_span_days != 1 else ''}")
                    o4.metric("Source", "Play Store")
                    st.caption(
                        f"📅 Reviews span **{result['date_min'].strftime('%Y-%m-%d')}** to "
                        f"**{result['date_max'].strftime('%Y-%m-%d')}**. Always check this before trusting "
                        f"a finding as \"current.\""
                    )
                    if date_span_days < days_window / 2:
                        st.warning(
                            f"⚠️ You requested a {days_window}-day window, but this app generates so many "
                            f"reviews that even the safety-capped sample only covers {date_span_days} day(s). "
                            f"Treat this as a recent snapshot, not a {days_window}-day trend view."
                        )

                    with st.expander("⚠️ Data limitations — read before using this report", expanded=True):
                        st.markdown("""
- **Single source** — Google Play Store only, no App Store/G2/Reddit yet.
- **Themes and analysis text are AI-generated live** for this specific run, grounded in the actual review quotes shown in each drill-down — but not independently human-validated the way our original Swiggy report was.
                        """)

                    st.markdown("### Top Opportunities")
                    top_n = result["opportunities"].head(8)
                    for i, row in top_n.iterrows():
                        render_opportunity(row, result["signals"], i + 1)

with st.expander("🔧 Can't find the right app? Enter its Play Store package ID directly"):
    st.caption(
        "Find this by searching the app on the Play Store website — the package ID is the part of the URL "
        "after `id=`, e.g. play.google.com/store/apps/details?id=**in.swiggy.android**"
    )
    manual_id = st.text_input("Package ID", placeholder="e.g. in.swiggy.android", key="manual_pkg")
    manual_name = st.text_input("Display name for this app", placeholder="e.g. Swiggy", key="manual_name")
    manual_analyze = st.button("✅ Analyze This Package ID")
    if manual_analyze and manual_id.strip():
        client = get_client()
        manual_app = {"title": manual_name.strip() or manual_id.strip(), "appId": manual_id.strip()}
        result = run_full_analysis(manual_name.strip() or manual_id.strip(), manual_app, days_window, client)
        if result:
            st.markdown(f"## {result['company']} — Product Intelligence Report")
            date_span_days = (result["date_max"] - result["date_min"]).days
            o1, o2, o3, o4 = st.columns(4)
            o1.metric("Signals analyzed", result["total_reviews"])
            o2.metric("Opportunities found", len(result["opportunities"]))
            o3.metric("Date range", f"{date_span_days} day{'s' if date_span_days != 1 else ''}")
            o4.metric("Source", "Play Store")
            st.caption(
                f"📅 Reviews span **{result['date_min'].strftime('%Y-%m-%d')}** to "
                f"**{result['date_max'].strftime('%Y-%m-%d')}**."
            )
            date_span_days = (result["date_max"] - result["date_min"]).days
            if date_span_days < days_window / 2:
                st.warning(
                    f"⚠️ You requested a {days_window}-day window, but this app generates so many "
                    f"reviews that even the safety-capped sample only covers {date_span_days} day(s). "
                    f"Treat this as a recent snapshot, not a {days_window}-day trend view."
                )
            with st.expander("⚠️ Data limitations — read before using this report", expanded=True):
                st.markdown("""
- **Single source** — Google Play Store only, no App Store/G2/Reddit yet.
- **Themes and analysis text are AI-generated live** for this specific run, grounded in the actual review quotes shown in each drill-down.
                """)
            st.markdown("### Top Opportunities")
            for i, row in result["opportunities"].head(8).iterrows():
                render_opportunity(row, result["signals"], i + 1)

if not st.session_state.candidates:
    st.markdown("""
    ### How this works
    1. Enter a company name and click **Find App** — confirm the right one from the matches shown
    2. Pick a time window that fits the company's review volume
    3. Click **Analyze** — SignalLens fetches, classifies, clusters, and scores the evidence, with every
       claim traceable back to a real review.

    **Note:** some companies (especially B2B SaaS) publish separate apps per product rather than one unified
    company app — pick the specific product you want analyzed. Some companies (especially dev tools) may have
    no meaningful consumer Play Store presence at all.
    """)
