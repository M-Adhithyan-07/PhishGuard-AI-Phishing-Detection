"""
PhishGuard — AI Phishing Detection Web App  (FIXED)
====================================================
Run:   python app.py
Open:  http://localhost:5000

═══════════════════════════════════════════════════════════════
BUGS FIXED IN THIS VERSION
═══════════════════════════════════════════════════════════════

BUG 1 — FEATURE CONTAMINATION (PRIMARY BUG — causes always-phishing)
-----------------------------------------------------------------------
The dataset_phishing.csv contains ~30 "page-content" columns that were
collected by actually fetching each URL during dataset creation:
    nb_hyperlinks, ratio_intHyperlinks, ratio_extHyperlinks,
    nb_extCSS, ratio_intMedia, ratio_extMedia, login_form,
    external_favicon, iframe, popup_window, sfh, safe_anchor,
    onmouseover, right_clic, empty_title, domain_in_title,
    domain_with_copyright, links_in_tags, submit_email, ...

Legitimate pages in the training data had REAL values for these
(e.g. nb_hyperlinks=87, ratio_intHyperlinks=0.72, …).
Phishing pages often had small/zero values.

At prediction time, extract_url_features() set ALL of these to 0
(since we don't fetch the page). So every URL looked like the
phishing class to the model.

FIX: Drop all page-content features before training. Train ONLY on
     the 57 structural/lexical URL features that we can actually
     compute at prediction time. Retrain and re-save the model.

BUG 2 — LABEL ENCODING NOT PERSISTED
----------------------------------------------------------------------
The original code reconstructed LabelEncoder on every run to check
the mapping, but never saved it. On some dataset variants the
label strings are "phishing"/"legitimate", on others they are
"0"/"1". We now detect and log the actual mapping at train time
and save it alongside the model so prediction always uses the
same mapping.

BUG 3 — NO CLASS BALANCING
----------------------------------------------------------------------
RandomForestClassifier without class_weight='balanced' overfits
to the majority class. Added class_weight='balanced'.

BUG 4 — WRONG PROBABILITY INDEX IN RISK SIGNALS
----------------------------------------------------------------------
The inline check `f.get("nb_redirections", …)` used a wrong key
name ('nb_redirections' vs 'nb_redirection'). Fixed.

═══════════════════════════════════════════════════════════════
"""

import os, re, time, json, traceback, urllib.parse
from pathlib import Path

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np

# ── lazy-load heavy deps ────────────────────────────────────────────────────
def _import_ml():
    import joblib
    import nltk
    from nltk.corpus import stopwords
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix
    return (joblib, nltk, stopwords, LogisticRegression,
            RandomForestClassifier, TfidfVectorizer,
            train_test_split, LabelEncoder,
            classification_report, confusion_matrix)

app = Flask(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
HERE         = Path(__file__).parent
BASE_DATASET = HERE / "phishing-attack-dataset"
MODEL_DIR    = HERE / "models"
MODEL_DIR.mkdir(exist_ok=True)

EMAIL_MODEL_PATH = MODEL_DIR / "email_sms_model.joblib"
EMAIL_VEC_PATH   = MODEL_DIR / "email_sms_vectorizer.joblib"
URL_MODEL_PATH   = MODEL_DIR / "url_model.joblib"
URL_COLS_PATH    = MODEL_DIR / "url_feature_cols.json"
URL_LABEL_PATH   = MODEL_DIR / "url_label_map.json"   # NEW — persisted label map

# ── Globals ──────────────────────────────────────────────────────────────────
email_model      = None
email_vectorizer = None
url_model        = None
url_feature_cols = None
url_label_map    = None   # dict: {"legitimate": int, "phishing": int}


# ════════════════════════════════════════════════════════════════════════════
# PAGE-CONTENT FEATURES — excluded from training & prediction
# because we cannot compute them without fetching the live page.
# Training on them caused the "always phishing" bug:
#   real URLs had non-zero values in the dataset;
#   at inference time we set them all to 0 → model saw phishing pattern.
# ════════════════════════════════════════════════════════════════════════════
PAGE_CONTENT_COLS = {
    "nb_hyperlinks", "ratio_intHyperlinks", "ratio_extHyperlinks",
    "ratio_nullHyperlinks", "nb_extCSS", "ratio_intRedirection",
    "ratio_extRedirection", "ratio_intErrors", "ratio_extErrors",
    "login_form", "external_favicon", "links_in_tags", "submit_email",
    "ratio_intMedia", "ratio_extMedia", "sfh", "iframe", "popup_window",
    "safe_anchor", "onmouseover", "right_clic", "empty_title",
    "domain_in_title", "domain_with_copyright", "whois_registered_domain",
    "domain_registration_length", "domain_age", "web_traffic",
    "dns_record", "google_index", "page_rank", "statistical_report",
}


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _safe_load_joblib(path):
    import joblib
    try:
        return joblib.load(path)
    except Exception:
        return None

def _find_sms_collection():
    folder = BASE_DATASET / "uci-sms-spam-collection-data-set"
    for name in ("SMSSpamCollection.txt", "SMSSpamCollection", "SMSSpamCollection.csv"):
        p = folder / name
        if p.exists():
            return p
    if folder.exists():
        files = list(folder.iterdir())
        if files:
            return files[0]
    return None


# ════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ════════════════════════════════════════════════════════════════════════════

def train_email_sms_model():
    (joblib, nltk, stopwords, LogisticRegression,
     RandomForestClassifier, TfidfVectorizer, train_test_split,
     LabelEncoder, classification_report, confusion_matrix) = _import_ml()

    nltk.download("stopwords", quiet=True)
    dfs = []

    simple_paths = [
        BASE_DATASET / "spam-sms-classification-using-nlp" / "Spam_SMS.csv",
        BASE_DATASET / "spam-email-classification" / "email.csv",
    ]
    for p in simple_paths:
        if p.exists():
            try:
                df = pd.read_csv(p)
                df = df.iloc[:, :2]
                df.columns = ["Category", "Message"]
                dfs.append(df)
                print(f"  ✔ {p.name}")
            except Exception as e:
                print(f"  ✘ {p.name}: {e}")

    uci = _find_sms_collection()
    if uci:
        try:
            df = pd.read_csv(uci, sep="\t", header=None, names=["Category", "Message"])
            dfs.append(df)
            print(f"  ✔ {uci.name}")
        except Exception as e:
            print(f"  ✘ SMSSpamCollection: {e}")

    phishing_names = ["CEAS_08.csv","Enron.csv","Ling.csv",
                      "Nigerian_Fraud.csv","SpamAssasin.csv","phishing_email.csv"]
    for name in phishing_names:
        p = BASE_DATASET / "phishing-email-dataset" / name
        if p.exists():
            try:
                df = pd.read_csv(p)
                df = df.iloc[:, :2]
                df.columns = ["Category", "Message"]
                dfs.append(df)
                print(f"  ✔ {name}")
            except Exception as e:
                print(f"  ✘ {name}: {e}")

    if not dfs:
        raise FileNotFoundError(
            "No email/SMS dataset files found.\n"
            f"Expected datasets inside: {BASE_DATASET}"
        )

    df = pd.concat(dfs, ignore_index=True)
    df.dropna(subset=["Category", "Message"], inplace=True)
    df["Category"] = df["Category"].astype(str).str.lower().str.strip()
    df = df[df["Category"].isin(["spam", "ham"])].copy()
    df["Category"] = df["Category"].map({"spam": 0, "ham": 1})
    df.dropna(inplace=True)

    print(f"\n  📊 Email/SMS class distribution:")
    print(f"     spam (0): {(df['Category']==0).sum()}")
    print(f"     ham  (1): {(df['Category']==1).sum()}")

    spam_df = df[df["Category"] == 0]
    ham_df  = df[df["Category"] == 1].sample(len(spam_df), random_state=42)
    df_bal  = pd.concat([spam_df, ham_df]).sample(frac=1, random_state=42)

    X, Y = df_bal["Message"], df_bal["Category"]
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=3)

    vec = TfidfVectorizer(stop_words="english", lowercase=True)
    X_tr = vec.fit_transform(X_train)
    X_te = vec.transform(X_test)

    mdl = LogisticRegression(max_iter=1000, solver="lbfgs")
    mdl.fit(X_tr, Y_train)

    acc = mdl.score(X_te, Y_te) if False else mdl.score(X_te, Y_test)
    print(f"\n  📈 Email/SMS model test accuracy: {acc:.4f}")
    preds = mdl.predict(X_te)
    print(classification_report(Y_test, preds, target_names=["spam","ham"]))

    joblib.dump(mdl, EMAIL_MODEL_PATH)
    joblib.dump(vec, EMAIL_VEC_PATH)
    print("  ✔ Email/SMS model saved.")
    return mdl, vec


def train_url_model():
    """
    Train URL model using ONLY structural/lexical features that can be
    computed from the URL string alone — no page-content features.
    This fixes the 'always predicts phishing' bug.
    """
    (joblib, nltk, stopwords, LogisticRegression,
     RandomForestClassifier, TfidfVectorizer, train_test_split,
     LabelEncoder, classification_report, confusion_matrix) = _import_ml()

    url_csv = BASE_DATASET / "dataset_phishing.csv"
    if not url_csv.exists():
        raise FileNotFoundError(
            f"URL dataset not found: {url_csv}\n"
            "Edit BASE_DATASET in app.py."
        )

    df = pd.read_csv(url_csv)
    print(f"\n  📋 URL dataset shape: {df.shape}")
    print(f"  📋 Columns: {list(df.columns[:10])} ...")

    # ── Drop URL string column ───────────────────────────────────────────────
    df.drop(columns=["url"], errors="ignore", inplace=True)

    # ── Inspect & encode label ───────────────────────────────────────────────
    print(f"\n  📊 Raw 'status' value counts:")
    print(df["status"].value_counts())

    le = LabelEncoder()
    df["status_encoded"] = le.fit_transform(df["status"])

    # Build and persist the label map so prediction uses same mapping
    label_map = {cls: int(idx) for idx, cls in enumerate(le.classes_)}
    print(f"\n  🔑 LabelEncoder class mapping: {label_map}")
    # Determine which encoded integer means 'legitimate'
    legit_key = None
    for k in label_map:
        if "legit" in k.lower():
            legit_key = k
            break
    if legit_key is None:
        # Fallback: whichever key is NOT 'phishing'
        for k in label_map:
            if "phish" not in k.lower():
                legit_key = k
                break
    legit_encoded = label_map.get(legit_key, 0)
    print(f"  ✔ Legitimate class → encoded as: {legit_encoded}")

    df.drop(columns=["status"], inplace=True)
    y = df["status_encoded"]

    # ── Drop page-content features (THE BUG FIX) ────────────────────────────
    cols_before = set(df.columns)
    drop_cols   = [c for c in df.columns if c in PAGE_CONTENT_COLS]
    df.drop(columns=drop_cols, errors="ignore", inplace=True)
    cols_after  = set(df.columns)
    print(f"\n  🗑  Dropped {len(drop_cols)} page-content features that cannot be")
    print(f"      computed at inference time (root cause of always-phishing bug):")
    for c in sorted(drop_cols):
        print(f"       - {c}")
    print(f"  ✔ Remaining structural features: {len(cols_after)-1}")  # -1 for status_encoded

    # ── Handle sentinel values ───────────────────────────────────────────────
    X = df.drop(columns=["status_encoded"])
    for col in ["domain_age", "domain_registration_length"]:
        if col in X.columns:
            med = X[X[col] != -1][col].median()
            X[col] = X[col].replace(-1, med)
            print(f"  ✔ Imputed sentinel -1 in '{col}' with median {med:.1f}")

    # ── Class distribution ───────────────────────────────────────────────────
    print(f"\n  📊 URL class distribution (encoded):")
    print(y.value_counts().to_string())
    imbalance_ratio = y.value_counts().max() / y.value_counts().min()
    print(f"  📊 Imbalance ratio: {imbalance_ratio:.2f}x")

    # ── Train / test split ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # ── Train with class balancing ───────────────────────────────────────────
    print(f"\n  🔄 Training RandomForest with class_weight='balanced'…")
    mdl = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced",   # FIX: handles class imbalance
        random_state=42,
        n_jobs=-1,
    )
    mdl.fit(X_train, y_train)

    # ── Evaluation ───────────────────────────────────────────────────────────
    y_pred = mdl.predict(X_test)
    acc    = (y_pred == y_test).mean()
    print(f"\n  📈 URL model test accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred,
                                target_names=le.classes_))
    cm = confusion_matrix(y_test, y_pred)
    print("  Confusion matrix:")
    print(cm)

    # ── Sanity check: predict a few well-known legit URLs ───────────────────
    print("\n  🧪 Sanity-check on well-known URLs:")
    _sanity_check(mdl, list(X.columns), label_map, legit_encoded)

    # ── Save ─────────────────────────────────────────────────────────────────
    feature_cols = list(X.columns)
    joblib.dump(mdl, URL_MODEL_PATH)
    with open(URL_COLS_PATH, "w") as f:
        json.dump(feature_cols, f)
    with open(URL_LABEL_PATH, "w") as f:
        json.dump({"label_map": label_map, "legit_encoded": legit_encoded}, f)

    print("  ✔ URL model saved.")
    return mdl, feature_cols, label_map, legit_encoded


def _sanity_check(mdl, feature_cols, label_map, legit_encoded):
    """Run a quick prediction on a few known-legit URLs after training."""
    test_urls = [
        "https://www.google.com",
        "https://github.com",
        "https://amazon.in",
        "http://paypa1-secure-login.xyz/verify/account?id=123456789",
    ]
    for url in test_urls:
        feats = extract_url_features_structural(url)
        row   = pd.DataFrame([feats])
        for col in feature_cols:
            if col not in row.columns:
                row[col] = 0
        row  = row[feature_cols]
        pred = int(mdl.predict(row)[0])
        prob = mdl.predict_proba(row)[0]
        tag  = "✅ LEGIT" if pred == legit_encoded else "🚨 PHISH"
        print(f"    {tag}  [{prob[legit_encoded]*100:.1f}% legit]  {url}")


# ════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_or_train_models():
    global email_model, email_vectorizer, url_model, url_feature_cols, url_label_map
    import joblib

    # ── Email / SMS ──────────────────────────────────────────────────────────
    em = _safe_load_joblib(EMAIL_MODEL_PATH)
    ev = _safe_load_joblib(EMAIL_VEC_PATH)
    if em is not None and ev is not None:
        email_model, email_vectorizer = em, ev
        print("  ✔ Email/SMS model loaded from disk.")
    else:
        print("  🔄 Training Email/SMS model…")
        try:
            email_model, email_vectorizer = train_email_sms_model()
        except Exception as e:
            print(f"  ✘ Email/SMS training failed: {e}")

    # ── URL ──────────────────────────────────────────────────────────────────
    # Force retrain if the old model was trained WITH page-content features
    # (i.e. url_feature_cols.json contains any PAGE_CONTENT_COLS keys)
    need_retrain = True
    um = _safe_load_joblib(URL_MODEL_PATH)
    if um is not None and URL_COLS_PATH.exists():
        with open(URL_COLS_PATH) as f:
            saved_cols = json.load(f)
        contaminated = [c for c in saved_cols if c in PAGE_CONTENT_COLS]
        if contaminated:
            print(f"\n  ⚠  Saved URL model was trained with {len(contaminated)} page-content")
            print(f"     features (root cause of always-phishing bug).")
            print(f"     Forcing retrain with structural-only features…")
        else:
            need_retrain = False

    if not need_retrain and URL_LABEL_PATH.exists():
        url_model = um
        with open(URL_COLS_PATH) as f:
            url_feature_cols = json.load(f)
        with open(URL_LABEL_PATH) as f:
            d = json.load(f)
            url_label_map = d
        print("  ✔ URL model loaded from disk.")
    else:
        print("  🔄 Training URL model (may take ~30-60 s)…")
        try:
            url_model, url_feature_cols, lmap, legit_enc = train_url_model()
            url_label_map = {"label_map": lmap, "legit_encoded": legit_enc}
        except Exception as e:
            print(f"  ✘ URL training failed: {e}")
            traceback.print_exc()


# ════════════════════════════════════════════════════════════════════════════
# URL FEATURE EXTRACTION — STRUCTURAL ONLY
# (removed all page-content features that require live fetching)
# ════════════════════════════════════════════════════════════════════════════

SHORTENING = {"bit.ly","goo.gl","tinyurl.com","ow.ly","t.co","tr.im",
              "is.gd","cli.gs","u.nu","url.ie","tiny.cc","qr.ae","adf.ly"}

PHISH_WORDS = ["secure","account","update","free","login","signin","bank",
               "verify","confirm","password","credential","click","winner",
               "prize","claim","urgent","webscr","ebayisapi"]

BRAND_LIST  = ["paypal","google","apple","amazon","microsoft","facebook",
               "netflix","instagram","twitter","ebay","chase","wellsfargo",
               "bankofamerica","citibank","dhl","fedex","ups","usps"]

SUSP_TLDS   = {".xyz",".top",".club",".online",".site",".live",".space",
               ".click",".link",".win",".loan",".download",".stream"}


def extract_url_features_structural(raw_url: str) -> dict:
    """
    Extract ONLY structural/lexical features computable from the URL string.
    Page-content features are intentionally omitted — they are excluded from
    the model entirely to prevent the 'always phishing' prediction bug.
    """
    url = raw_url.strip()
    try:
        parsed = urllib.parse.urlparse(url if "://" in url else "http://" + url)
    except Exception:
        parsed = urllib.parse.urlparse("http://unknown.com")

    scheme   = parsed.scheme or ""
    hostname = (parsed.hostname or "").lower()
    path     = parsed.path or ""
    query    = parsed.query or ""
    full     = url

    def cnt(s, c): return s.count(c)
    def words(s):  return [w for w in re.split(r"[.\-/=?&_~%+@#!]", s) if w]

    digits_url  = sum(c.isdigit() for c in full)
    digits_host = sum(c.isdigit() for c in hostname)

    wr = words(full); wh = words(hostname); wp = words(path)

    parts      = hostname.split(".")
    tld        = ("." + parts[-1]) if parts else ""
    nb_subdoms = max(len(parts) - 2, 0)

    return {
        "length_url":               len(full),
        "length_hostname":          len(hostname),
        "ip":                       1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", hostname) else 0,
        "nb_dots":                  cnt(full, "."),
        "nb_hyphens":               cnt(full, "-"),
        "nb_at":                    cnt(full, "@"),
        "nb_qm":                    cnt(full, "?"),
        "nb_and":                   cnt(full, "&"),
        "nb_or":                    cnt(full, "|"),
        "nb_eq":                    cnt(full, "="),
        "nb_underscore":            cnt(full, "_"),
        "nb_tilde":                 cnt(full, "~"),
        "nb_percent":               cnt(full, "%"),
        "nb_slash":                 cnt(full, "/"),
        "nb_star":                  cnt(full, "*"),
        "nb_colon":                 cnt(full, ":"),
        "nb_comma":                 cnt(full, ","),
        "nb_semicolumn":            cnt(full, ";"),
        "nb_dollar":                cnt(full, "$"),
        "nb_space":                 cnt(full, " ") + cnt(full, "%20"),
        "nb_www":                   cnt(full.lower(), "www"),
        "nb_com":                   cnt(full.lower(), ".com"),
        "nb_dslash":                cnt(full, "//"),
        "http_in_path":             1 if "http" in path.lower() else 0,
        "https_token":              1 if scheme == "https" else 0,
        "ratio_digits_url":         digits_url / max(len(full), 1),
        "ratio_digits_host":        digits_host / max(len(hostname), 1),
        "punycode":                 1 if "xn--" in hostname else 0,
        "port":                     1 if parsed.port else 0,
        "tld_in_path":              1 if tld in path else 0,
        "tld_in_subdomain":         1 if any(tld in p for p in parts[:-2]) else 0,
        "abnormal_subdomain":       1 if nb_subdoms > 3 else 0,
        "nb_subdomains":            nb_subdoms,
        "prefix_suffix":            1 if "-" in hostname else 0,
        "random_domain":            1 if re.search(r"[0-9]{3,}", hostname) else 0,
        "shortening_service":       1 if hostname in SHORTENING else 0,
        "path_extension":           1 if re.search(r"\.\w{2,4}$", path) else 0,
        "nb_redirection":           path.count("//"),
        "nb_external_redirection":  cnt(query, "http"),
        "length_words_raw":         sum(len(w) for w in wr),
        "char_repeat":              max((full.count(c) for c in set(full)), default=0),
        "shortest_words_raw":       min((len(w) for w in wr), default=0),
        "shortest_word_host":       min((len(w) for w in wh), default=0),
        "shortest_word_path":       min((len(w) for w in wp), default=0),
        "longest_words_raw":        max((len(w) for w in wr), default=0),
        "longest_word_host":        max((len(w) for w in wh), default=0),
        "longest_word_path":        max((len(w) for w in wp), default=0),
        "avg_words_raw":            float(np.mean([len(w) for w in wr])) if wr else 0.0,
        "avg_word_host":            float(np.mean([len(w) for w in wh])) if wh else 0.0,
        "avg_word_path":            float(np.mean([len(w) for w in wp])) if wp else 0.0,
        "phish_hints":              sum(h in full.lower() for h in PHISH_WORDS),
        "domain_in_brand":          1 if any(b in hostname for b in BRAND_LIST) else 0,
        "brand_in_subdomain":       1 if any(b in ".".join(parts[:-2]) for b in BRAND_LIST) else 0,
        "brand_in_path":            1 if any(b in path.lower() for b in BRAND_LIST) else 0,
        "suspecious_tld":           1 if tld in SUSP_TLDS else 0,
    }


# Keep old name as alias for backward compat with any code that imports it
extract_url_features = extract_url_features_structural


# ════════════════════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/api/check", methods=["POST", "OPTIONS"])
def check():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    try:
        data       = request.get_json(force=True, silent=True) or {}
        check_type = data.get("type", "").strip()
        text       = data.get("text", "").strip()

        if not text:
            return jsonify({"error": "Input is empty."}), 400
        if check_type not in ("url", "email", "sms"):
            return jsonify({"error": f"Unknown type: '{check_type}'"}), 400

        start = time.time()

        # ── Email / SMS ──────────────────────────────────────────────────────
        if check_type in ("email", "sms"):
            if email_model is None or email_vectorizer is None:
                return jsonify({"error": "Email/SMS model not loaded."}), 503

            features  = email_vectorizer.transform([text])
            pred      = int(email_model.predict(features)[0])
            proba     = email_model.predict_proba(features)[0].tolist()
            # Email model: 0=spam, 1=ham  (set explicitly in train_email_sms_model)
            spam_prob = round(proba[0] * 100, 2)
            ham_prob  = round(proba[1] * 100, 2)
            safe      = pred == 1
            elapsed   = round((time.time() - start) * 1000, 2)

            print(f"\n  [EMAIL/SMS DEBUG]")
            print(f"    pred={pred}  spam_prob={spam_prob}%  ham_prob={ham_prob}%  safe={safe}")

            return jsonify({
                "type":       check_type,
                "label":      "Legitimate Message" if safe else "Spam / Phishing",
                "safe":       safe,
                "confidence": ham_prob if safe else spam_prob,
                "spam_prob":  spam_prob,
                "ham_prob":   ham_prob,
                "elapsed_ms": elapsed,
            })

        # ── URL ──────────────────────────────────────────────────────────────
        if url_model is None or url_feature_cols is None:
            return jsonify({"error": "URL model not loaded."}), 503

        # Determine which class index == legitimate
        legit_encoded = url_label_map.get("legit_encoded", 0) if url_label_map else 0
        label_map     = url_label_map.get("label_map", {})    if url_label_map else {}

        feats_dict = extract_url_features_structural(text)

        # Align to training columns exactly
        row = pd.DataFrame([feats_dict])
        for col in url_feature_cols:
            if col not in row.columns:
                row[col] = 0
        row = row[url_feature_cols]

        pred    = int(url_model.predict(row)[0])
        proba   = url_model.predict_proba(row)[0].tolist()

        # Use the saved class ordering from training
        classes = url_model.classes_.tolist()
        # proba[i] corresponds to classes[i]
        legit_idx = classes.index(legit_encoded) if legit_encoded in classes else 0
        phish_idx = 1 - legit_idx

        legit_p = round(proba[legit_idx] * 100, 2)
        phish_p = round(proba[phish_idx] * 100, 2)
        safe    = (pred == legit_encoded)
        elapsed = round((time.time() - start) * 1000, 2)

        # ── Debug output ─────────────────────────────────────────────────────
        print(f"\n  [URL DEBUG] Input: {text[:80]}")
        print(f"    classes={classes}  legit_encoded={legit_encoded}")
        print(f"    pred={pred}  legit_p={legit_p}%  phish_p={phish_p}%  safe={safe}")
        print(f"    label_map={label_map}")
        print(f"    Key features: https={feats_dict['https_token']}  "
              f"ip={feats_dict['ip']}  "
              f"subdoms={feats_dict['nb_subdomains']}  "
              f"phish_hints={feats_dict['phish_hints']}  "
              f"susp_tld={feats_dict['suspecious_tld']}")

        # ── Risk signals ─────────────────────────────────────────────────────
        f = feats_dict
        signals = []
        if f["ip"]:
            signals.append("IP address used as hostname (bypasses domain reputation checks)")
        if f["shortening_service"]:
            signals.append("URL shortening service detected — destination is hidden")
        if f["prefix_suffix"]:
            signals.append("Hyphen in domain name (common in look-alike / typosquat domains)")
        if not f["https_token"]:
            signals.append("No HTTPS — connection is unencrypted")
        if f["nb_subdomains"] > 3:
            signals.append(f"Excessive subdomains ({f['nb_subdomains']}) — typical phishing structure")
        if f["phish_hints"] > 0:
            signals.append(f"{f['phish_hints']} phishing keyword(s) found in URL path or query")
        if f["suspecious_tld"]:
            signals.append("Suspicious top-level domain (e.g. .xyz, .top, .click)")
        if f["tld_in_subdomain"]:
            signals.append("TLD embedded in subdomain — brand spoofing tactic")
        if f["brand_in_subdomain"]:
            signals.append("Trusted brand name used in subdomain to deceive users")
        if f["nb_at"] > 0:
            signals.append("@ symbol in URL — browser ignores everything before the @")
        if f["length_url"] > 100:
            signals.append(f"Unusually long URL ({f['length_url']} characters)")
        if f["nb_hyphens"] > 4:
            signals.append(f"High hyphen count ({f['nb_hyphens']}) — look-alike domain indicator")
        if f["http_in_path"]:
            signals.append("'http' found inside URL path — possible redirect chain")
        if f["punycode"]:
            signals.append("Punycode (xn--) in hostname — homograph / international spoof attack")
        if f["nb_redirection"] > 0:
            signals.append("URL contains redirect sequences")

        return jsonify({
            "type":         "url",
            "label":        "Legitimate URL" if safe else "Phishing URL",
            "safe":         safe,
            "confidence":   legit_p if safe else phish_p,
            "phish_prob":   phish_p,
            "legit_prob":   legit_p,
            "elapsed_ms":   elapsed,
            "risk_signals": signals,
            "features": {
                "URL Length":          f["length_url"],
                "Hostname Length":     f["length_hostname"],
                "HTTPS":               "Yes" if f["https_token"] else "No",
                "IP as Hostname":      "Yes" if f["ip"] else "No",
                "Subdomains":          f["nb_subdomains"],
                "Hyphens in Domain":   f["nb_hyphens"],
                "Phishing Keywords":   f["phish_hints"],
                "Suspicious TLD":      "Yes" if f["suspecious_tld"] else "No",
                "URL Shortener":       "Yes" if f["shortening_service"] else "No",
                "Punycode":            "Yes" if f["punycode"] else "No",
            },
        })

    except Exception:
        tb = traceback.format_exc()
        print("⚠ /api/check error:\n", tb)
        return jsonify({"error": "Internal server error.", "detail": tb}), 500


@app.route("/api/status")
def status():
    return jsonify({
        "email_model":    email_model is not None,
        "url_model":      url_model is not None,
        "dataset_path":   str(BASE_DATASET),
        "dataset_exists": BASE_DATASET.exists(),
        "url_label_map":  url_label_map,
    })


# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "═" * 56)
    print("  PhishGuard — AI Phishing Detection System  (FIXED)")
    print("═" * 56)
    print(f"  Dataset folder : {BASE_DATASET}")
    print(f"  Dataset exists : {BASE_DATASET.exists()}")
    print("═" * 56)
    print("  Loading / training models…")
    load_or_train_models()
    print("═" * 56)
    print("  Open browser →  http://localhost:5000")
    print("═" * 56 + "\n")
    import webbrowser
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=False, host="0.0.0.0", port=5000)