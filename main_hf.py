import streamlit as st
import pandas as pd
from tqdm.auto import tqdm
from lexicons import CHAR_NORMALIZATION, BN_EN_MAP, PHRASES, BANGLISH_WORDS, NEGATIONS, RELIGIOUS_POSITIVE, load_bnlex, CANONICAL_WORDS, INTENSIFIER_SET, BNLEX
from emoji_dicts import POSITIVE_EMOJIS, NEGATIVE_EMOJIS
import re, emoji, matplotlib.pyplot as plt, seaborn as sns
from transformers import pipeline
from io import BytesIO
from rapidfuzz import process, fuzz

st.set_page_config(page_title="Bangla+English Sentiment Analyzer", layout="wide")
tqdm.pandas()

# ----------------- Load Bangla Lexicons (Lazy) -----------------
@st.cache_resource
def load_bnlex_lazy():
    return load_bnlex("Bangla-Vulgar-Lexicon/vulgar.txt",
                      "BNLexicon/unique_list_positive.txt",
                      "BNLexicon/unique_list_negative.txt", BNLEX)

bnlex, vulgar_words = load_bnlex_lazy()

# ----------------- Canonical words set -----------------
ALL_CANONICAL = CANONICAL_WORDS["positive"] + CANONICAL_WORDS["negative"] + list(BANGLISH_WORDS.keys()) + list(RELIGIOUS_POSITIVE)
ALL_CANONICAL_SET = (
    set(ALL_CANONICAL)
    | set(BANGLISH_WORDS)
    | set(RELIGIOUS_POSITIVE)
)

BANGLISH_SET = set(BANGLISH_WORDS)| set(RELIGIOUS_POSITIVE)

def fuzzy_correct_word(word, threshold=85):
    if word in ALL_CANONICAL_SET:
        return word
    match = process.extractOne(
        word,
        ALL_CANONICAL,
        scorer=fuzz.ratio
    )
    if match and match[1] >= threshold:
        return match[0]
    return word

def fuzzy_bangla_word(word, threshold=85):
    if word in BANGLISH_SET:
        return word
    match = process.extractOne(
        word,
        BANGLISH_SET,
        scorer=fuzz.ratio
    )
    if match and match[1] >= threshold:
        return match[0]
    return word

def is_transliteration(text):
    tokens = text.split()
    corrected = [fuzzy_bangla_word(w) for w in tokens]

    banglish_hits = sum(
        (w in BANGLISH_WORDS) 
        or (w in bnlex) 
        or (w in RELIGIOUS_POSITIVE)
        for w in corrected
    )
    return banglish_hits >= 1


# ----------------- Preprocessing -----------------
def normalize_bangla_chars(text):
    if not isinstance(text,str): return ""
    for s,t in CHAR_NORMALIZATION.items(): text = text.replace(s,t)
    return text

def normalize_whitespace(text):
    if not isinstance(text,str): return ""
    text = text.replace("\u200b","").replace("\xa0"," ").strip()
    text = re.sub(r"\s+"," ", text)
    return text


def replace_bn_en_words(text):
    tokens = text.split()
    corrected = [fuzzy_correct_word(w) for w in tokens]  # just fix spelling
    replaced = [BN_EN_MAP.get(w, w) for w in corrected]  # map to English
    return " ".join(replaced)



def separate_emojis(text):
    if not isinstance(text, str): 
        return ""
    # space around emoji
    return emoji.replace_emoji(text, replace=lambda e, _: f" {e} ")


def preprocess_text(text):
    if not isinstance(text,str): return ""
    original = text
    text = text.lower()
    text = re.sub(r"http\S+|www\S+","",text)
    text = normalize_whitespace(text)
    text = text.replace("."," ")
    text = text.replace(","," ")
    text = separate_emojis(text)
    # text = replace_bn_en_words(text)
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r"[^\w\s\u0980-\u09FF]", "", text)
    return text

# ----------------- English Model (Hugging Face Hub) -----------------
@st.cache_resource
def load_english_model():
    return pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment"
    )

english_model = load_english_model()
def english_sentiment(text):
    label = english_model(text[:512])[0]["label"]
    if label in ("4 stars","5 stars"): return "positive"
    elif label=="3 stars": return "neutral"
    return "negative"

# ----------------- Emoji scoring -----------------
def emoji_sentiment_score(text):
    if not isinstance(text, str):
        return 0

    emoji_list = emoji.emoji_list(text)
    extracted = [e["emoji"] for e in emoji_list]

    pos_count = sum(e in POSITIVE_EMOJIS for e in extracted)
    neg_count = sum(e in NEGATIVE_EMOJIS for e in extracted)
    score=pos_count - neg_count
    return score

# ----------------- helper functions -----------------
def has_negative_suffix(word):
    return word.endswith("না")

def is_intensifier(word, threshold=90):
    if word in INTENSIFIER_SET:
        return True
    match = process.extractOne(word, INTENSIFIER_SET, scorer=fuzz.ratio)
    return match and match[1] >= threshold

def match_religious_word(word, threshold=85):
    # Exact match
    if word in RELIGIOUS_POSITIVE:
        return True
    
    # Fuzzy match
    match = process.extractOne(word, RELIGIOUS_POSITIVE, scorer=fuzz.ratio)
    if match and match[1] >= threshold:
        return True
    
    return False


# ----------------- Lexicon lookup -----------------
ALL_LEXICON_WORDS = list(bnlex.keys()) + list(BANGLISH_WORDS.keys())

def lexicon_lookup(word, threshold=85):
    # word_norm = normalize_bangla_chars(word)
    score = bnlex.get(word,0) + BANGLISH_WORDS.get(word,0)
    found_exact = word in bnlex or word in BANGLISH_WORDS

    if score != 0:
        return score
    
    # Skip fuzzy matching for multi-word phrases
    if len(word.split()) > 1:
        return 0
    
    if not found_exact:
        match = process.extractOne(word, ALL_LEXICON_WORDS, scorer=fuzz.ratio)
        if match and match[1] >= threshold:
            matched_word = match[0]
            return bnlex.get(matched_word,0) + BANGLISH_WORDS.get(matched_word,0)
    return 0

def lexicon_sentiment(text, original_text):
    score = 0
    for phrase, val in PHRASES.items():
        if phrase in text: 
            score += val
    
    skip_next = False
    words=text.split()
    for i,w in enumerate(words):
        if skip_next:
            skip_next = False
            continue
        if is_intensifier(w) and i + 1 < len(words):
            next_word = words[i + 1]
            next_score = lexicon_lookup(next_word)

            if next_score > 0:
                score += next_score * 2
            elif next_score < 0:
                score += next_score * 2
            skip_next = True
            continue
        
        base = lexicon_lookup(w)
        if has_negative_suffix(w) and w not in bnlex:
            # Strong penalty for suffix negation
            score -= 2

        if w in NEGATIONS and i+1 < len(text.split()):
            next_score = lexicon_lookup(words[i+1])
            if next_score != 0:
                score -= abs(next_score)
            else:
                score += base
        else: 
            score += base
            
        if match_religious_word(w):
            score += 2

        if w in vulgar_words: 
            score -= 2
    return score

def detect_script(text):
    if re.search(r"[\u0980-\u09FF]", text): return "bangla"
    if re.search(r"[a-zA-Z]", text): return "english"
    return "other"

# ----------------- Hybrid Sentiment (use preprocessed text) -----------------
def hybrid_sentiment(clean_text, original_text):
    emoji_score = emoji_sentiment_score(original_text)
    script = detect_script(clean_text)
    lex_score=lexicon_sentiment(clean_text, original_text)
    if script=="bangla":
        score = lex_score + emoji_score
        if score>=1: 
            return {"final_sentiment":"positive"}
        elif score<=-1: 
            return {"final_sentiment":"negative"}
        return {"final_sentiment":"neutral"}
    if script=="english":
        if is_transliteration(clean_text):
            total_score = lex_score + emoji_score
            if total_score >= 1:
                return {"final_sentiment": "positive"}
            elif total_score <= -1:
                return {"final_sentiment": "negative"}
            return {"final_sentiment": "neutral"}
        else:
            if not clean_text.strip() and emoji_score!=0:
                return {"final_sentiment":"positive" if emoji_score>0 else "negative"}
            label = english_sentiment(clean_text)
            if emoji_score>0 and label=='positive': 
                return {"final_sentiment":"positive"}
            if emoji_score<0 and label=='negative': 
                return {"final_sentiment":"negative"}
            return {"final_sentiment":label}
    if emoji_score>0: 
        return {"final_sentiment":"positive"}
    if emoji_score<0: 
        return {"final_sentiment":"negative"}
    return {"final_sentiment":"neutral"}

# ----------------------------- Streamlit GUI -----------------------------
st.title("Sentiment Analysis tool")

mode = st.radio(
    "Select Mode:",
    ["Compare with Agent Sentiment","Predict Sentiment Only"],
    horizontal=True
)

st.markdown("Upload a CSV file with a **`Customer Comment`** column. The tool will compute sentiment and compare with agent-labeled sentiment.")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.replace('\ufeff','')
    progress_bar = st.progress(0)  # initialize

    # ---------------- Preprocess all text once ----------------
    if mode == "Compare with Agent Sentiment":
        if "Customer Comment" not in df.columns or "Sentiment" not in df.columns:
            st.error("CSV must contain `Customer Comment` and `Sentiment` columns.")
            st.stop()
        st.info(f"Processing {len(df)} comments...")
        
        df["clean_text"] = df["Customer Comment"].progress_apply(preprocess_text)
        
        results = []
        for i, (_, row) in enumerate(df.iterrows(), 1):
            results.append(
                hybrid_sentiment(
                    row["clean_text"],
                    row["Customer Comment"]
                )["final_sentiment"]
            )
        progress_bar.progress(i / len(df))

        df["predicted_sentiment"] = results

        # The rest remains same
        df["Sentiment_clean"] = df["Sentiment"].str.strip().str.lower()
        df["Predicted_Sentiment_clean"] = df["predicted_sentiment"].str.strip().str.lower()
        df["Sentiment_Mismatch"] = df["Sentiment_clean"] != df["Predicted_Sentiment_clean"]
        st.success("✅ Sentiment comparison complete.")

        # Plotting
        agent_dist = df["Sentiment_clean"].value_counts(normalize=True).mul(100).reindex(["positive","neutral","negative"])
        pred_dist = df["Predicted_Sentiment_clean"].value_counts(normalize=True).mul(100).reindex(["positive","neutral","negative"])
        plot_df = pd.DataFrame({"Agent Marked": agent_dist,"Predicted": pred_dist}).reset_index().rename(columns={"index":"Sentiment"})
        plot_df_melted = plot_df.melt(id_vars="Sentiment", var_name="Source", value_name="Percentage")
        fig, ax = plt.subplots(figsize=(10,6))
        sns.barplot(data=plot_df_melted,x="Sentiment",y="Percentage",hue="Source",palette=["#2E86C1","#E74C3C"],ax=ax)
        ax.set_ylim(0,100)
        ax.set_title("Sentiment Distribution (%) — Agent vs Predicted")
        for container in ax.containers: ax.bar_label(container, fmt="%.1f%%", padding=3)
        st.pyplot(fig)

        st.write(f"**Number of mismatched rows:** {df['Sentiment_Mismatch'].sum()}")
        st.write(f"**Mismatched Percentage:** {(df['Sentiment_Mismatch'].mean() * 100):.2f}%")



        # Ensure required columns exist

        required_cols = ["Date", "Customer Comment", "Sentiment", 
                        "predicted_sentiment", "Sentiment_Mismatch"]

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required column(s): {missing_cols}")
            st.stop()

        final_df = df[required_cols].copy()

        # Optional: rename column to lowercase as requested
        final_df = final_df.rename(columns={"Sentiment_Mismatch": "sentiment_mismatch"})


    elif mode == "Predict Sentiment Only":
        if "text" not in df.columns:
            st.error("CSV must contain a `text` column.")
            st.stop()
        st.info(f"Processing {len(df)} comments...")
        df["clean_text"] = df["text"].progress_apply(preprocess_text)
        print(r.split for r in df["clean_text"])
        results = []
        for i, row in enumerate(df.itertuples(), 1):
            results.append(hybrid_sentiment(row.clean_text, row._asdict()["text"])["final_sentiment"])
            progress_bar.progress(i/len(df))
        df["predicted_sentiment"] = results
        st.success("✅ Sentiment prediction complete.")

        pred_dist = df["predicted_sentiment"].value_counts(normalize=True).mul(100)
        fig, ax = plt.subplots(figsize=(8,5))
        sns.barplot(x=pred_dist.index,y=pred_dist.values,palette="viridis",ax=ax)
        ax.set_ylim(0,100)
        ax.set_title("Predicted Sentiment Distribution (%)")
        for i, v in enumerate(pred_dist.values): ax.text(i, v+1, f"{v:.1f}%", ha='center')
        st.pyplot(fig)
        final_df = df[["date","text","predicted_sentiment"]].copy()

    # ----------------------------- Download -----------------------------
    ILLEGAL_EXCEL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
    def clean_for_excel(text):
        if not isinstance(text,str): return text
        return ILLEGAL_EXCEL_RE.sub("", text)

    for col in final_df.select_dtypes(include="object").columns:
        final_df[col] = final_df[col].apply(clean_for_excel)

    output = BytesIO()
    final_df.to_excel(output, index=False, engine="openpyxl")
    output.seek(0)

    st.download_button(
        label="Download Results as Excel",
        data=output,
        file_name="hybrid_sentiment_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
