import os
import re

import numpy as np
import pandas as pd
import streamlit as st

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi


# =========================================================
# إعدادات الصفحة
# =========================================================

st.set_page_config(
    page_title="محرك البحث الذكي للمحاضرات",
    page_icon="🔎",
    layout="wide"
)


# =========================================================
# إعدادات
# =========================================================

EXCEL_FILE = "Book.xlsx"

DESCRIPTION_COL = (
    "وصف المحاضرة \n"
    "او \n"
    "الدورة (في حال عدم وجود وصف للمحاضرات)"
)

E5_MODEL = "intfloat/multilingual-e5-base"
BGE_MODEL = "BAAI/bge-reranker-v2-m3"


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
    <style>

    .block-container {
        direction: rtl;
        text-align: right;
        max-width: 1100px;
        padding-top: 3rem;
    }

    .main-title {
        text-align: center;
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 30px;
    }

    .search-label {
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .result-count {
        font-size: 17px;
        color: #667085;
        margin: 25px 0 15px 0;
    }

    # .result-card {
    #     direction: rtl;
    #     text-align: right;
    #     background: white;
    #     border: 1px solid #e4e7ec;
    #     border-radius: 18px;
    #     padding: 24px;
    #     margin: 18px 0;
    #     box-shadow: 0 4px 15px rgba(16, 24, 40, 0.07);
    # }

    # .course-name {
    #     font-size: 14px;
    #     color: #667085;
    #     margin-bottom: 8px;
    # }

    # .lecture-name {
    #     font-size: 24px;
    #     font-weight: 700;
    #     color: #172033;
    #     margin-bottom: 18px;
    # }

    # .info-box {
    #     background: #f8fafc;
    #     border-radius: 12px;
    #     padding: 15px;
    #     margin-bottom: 15px;
    #     line-height: 1.9;
    #     color: #344054;
    # }

    # .score-box {
    #     display: inline-block;
    #     background: #eff6ff;
    #     border-radius: 10px;
    #     padding: 8px 14px;
    #     color: #2563eb;
    #     font-weight: 700;
    #     font-size: 17px;
    #     margin-bottom: 15px;
    # }

    # .lecture-button {
    #     display: inline-block;
    #     background: #2563eb;
    #     color: white !important;
    #     padding: 11px 20px;
    #     border-radius: 10px;
    #     text-decoration: none !important;
    #     font-weight: 600;
    #     font-size: 15px;
    # }

    .lecture-button:hover {
        background: #1d4ed8;
        color: white !important;
    }

    .no-results {
        direction: rtl;
        text-align: center;
        background: #f8f9fa;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 30px;
        margin-top: 25px;
        color: #667085;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# تنظيف النص
# =========================================================

def clean_text(value):

    if pd.isna(value):
        return ""

    return str(value).replace("\n", " ").strip()


def normalize_text(text):

    text = str(text).lower().strip()

    # إزالة التشكيل العربي
    text = re.sub(
        r"[\u064B-\u065F\u0670]",
        "",
        text
    )

    # توحيد الحروف العربية
    text = text.replace("أ", "ا")
    text = text.replace("إ", "ا")
    text = text.replace("آ", "ا")
    text = text.replace("ة", "ه")
    text = text.replace("ى", "ي")

    # إزالة الرموز الزائدة
    text = re.sub(
        r"[^\w\s\u0600-\u06FF]",
        " ",
        text
    )

    # توحيد المسافات
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# =========================================================
# تحميل Excel
# =========================================================

@st.cache_data
def load_excel():

    if not os.path.exists(EXCEL_FILE):

        raise FileNotFoundError(
            f"لم يتم العثور على ملف {EXCEL_FILE}"
        )

    return pd.read_excel(EXCEL_FILE)


# =========================================================
# تجهيز البيانات
# =========================================================

@st.cache_data
def prepare_dataframe(df):

    required_columns = [
        "ID",
        "اسم الدورة",
        "اسم المحاضرة",
        "التصنيف الأساسي",
        "التصنيف الفرعي",
        "keywords",
        DESCRIPTION_COL,
        "الرابط من المصدر الاساسي (حفظا لحقوق النشر)"
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "الأعمدة التالية غير موجودة في ملف Excel:\n"
            + "\n".join(missing_columns)
        )

    df = df.copy()

    df["lecture_clean"] = (
        df["اسم المحاضرة"]
        .fillna("")
        .astype(str)
        .apply(clean_text)
    )

    df["main_category_clean"] = (
        df["التصنيف الأساسي"]
        .fillna("")
        .astype(str)
        .apply(clean_text)
    )

    df["subcategory_clean"] = (
        df["التصنيف الفرعي"]
        .fillna("")
        .astype(str)
        .apply(clean_text)
    )

    df["keywords_clean"] = (
        df["keywords"]
        .fillna("")
        .astype(str)
        .apply(clean_text)
    )

    df["description_clean"] = (
        df[DESCRIPTION_COL]
        .fillna("")
        .astype(str)
        .apply(clean_text)
    )

    # =====================================================
    # Search Profile
    # =====================================================

    def build_search_profile(row):

        return f"""
اسم الدورة: {row["اسم الدورة"]}

اسم المحاضرة: {row["lecture_clean"]}

التصنيف الأساسي: {row["main_category_clean"]}

التصنيف الفرعي: {row["subcategory_clean"]}

الكلمات المفتاحية: {row["keywords_clean"]}

وصف المحاضرة: {row["description_clean"]}
""".strip()

    df["search_profile"] = df.apply(
        build_search_profile,
        axis=1
    )

    # =====================================================
    # BM25
    # =====================================================

    bm25_columns = [
        "اسم المحاضرة",
        "التصنيف الأساسي",
        "التصنيف الفرعي",
        "keywords"
    ]

    for col in bm25_columns:

        df[col + "_normalized"] = (
            df[col]
            .fillna("")
            .astype(str)
            .apply(normalize_text)
        )

    df["bm25_text"] = (
        df["اسم المحاضرة_normalized"]
        + " "
        + df["التصنيف الأساسي_normalized"]
        + " "
        + df["التصنيف الفرعي_normalized"]
        + " "
        + df["keywords_normalized"]
    )

    bm25_corpus = [
        text.split()
        for text in df["bm25_text"]
    ]

    return df, bm25_corpus


# =========================================================
# تحميل E5
# =========================================================

@st.cache_resource
def load_search_model():

    return SentenceTransformer(
        E5_MODEL
    )


# =========================================================
# تحميل BGE
# =========================================================

@st.cache_resource
def load_reranker():

    return CrossEncoder(
        BGE_MODEL,
        max_length=512
    )


# =========================================================
# إنشاء Embeddings
# =========================================================

@st.cache_data
def create_document_embeddings(
    search_profiles,
    _search_model
):

    return _search_model.encode(
        [
            "passage: " + text
            for text in search_profiles
        ],
        normalize_embeddings=True,
        show_progress_bar=False
    )


# =========================================================
# إنشاء BM25
# =========================================================

@st.cache_resource
def build_bm25(corpus):

    return BM25Okapi(corpus)


# =========================================================
# تحميل كل مكونات البحث
# =========================================================

try:

    df = load_excel()

    df, bm25_corpus = prepare_dataframe(
        df
    )

    search_model = load_search_model()

    document_embeddings = create_document_embeddings(
        tuple(df["search_profile"].tolist()),
        search_model
    )

    bm25 = build_bm25(
        bm25_corpus
    )

    reranker = load_reranker()

except Exception as e:

    st.error(
        "حدث خطأ أثناء تحميل النظام."
    )

    st.exception(e)

    st.stop()


# =========================================================
# Hybrid Search
# =========================================================

def search_library(
    query,
    max_candidates=30
):

    query = str(query).strip()

    if not query:
        return pd.DataFrame()

    normalized_query = normalize_text(
        query
    )

    query_tokens = normalized_query.split()

    if not query_tokens:
        return pd.DataFrame()

    # =====================================================
    # BM25
    # =====================================================

    bm25_scores = bm25.get_scores(
        query_tokens
    )

    bm25_order = np.argsort(
        -bm25_scores
    )

    # =====================================================
    # E5
    # =====================================================

    query_embedding = search_model.encode(
        ["query: " + query],
        normalize_embeddings=True
    )[0]

    e5_scores = np.dot(
        document_embeddings,
        query_embedding
    )

    e5_order = np.argsort(
        -e5_scores
    )

    # =====================================================
    # RRF
    # =====================================================

    n = len(df)

    bm25_rank = np.empty(
        n,
        dtype=int
    )

    e5_rank = np.empty(
        n,
        dtype=int
    )

    bm25_rank[bm25_order] = np.arange(
        1,
        n + 1
    )

    e5_rank[e5_order] = np.arange(
        1,
        n + 1
    )

    rrf_k = 60

    rrf_scores = (
        1 / (rrf_k + bm25_rank)
        +
        1 / (rrf_k + e5_rank)
    )

    # =====================================================
    # Candidate selection
    # =====================================================

    candidates = df.copy()

    candidates["bm25_score"] = bm25_scores
    candidates["e5_score"] = e5_scores
    candidates["rrf_score"] = rrf_scores

    candidates = (
        candidates
        .sort_values(
            "rrf_score",
            ascending=False
        )
        .head(max_candidates)
        .reset_index(drop=True)
    )

    # =====================================================
    # BGE Reranking
    # =====================================================

    pairs = [
        [
            query,
            row["search_profile"]
        ]
        for _, row in candidates.iterrows()
    ]

    candidates["bge_score"] = reranker.predict(
        pairs,
        show_progress_bar=False
    )

    # =====================================================
    # Direct topic matching
    # =====================================================

    direct_scores = []
    matched_terms_list = []

    query_word_set = set(
        query_tokens
    )

    for _, row in candidates.iterrows():

        searchable_text = " ".join([
            str(row["اسم المحاضرة"]),
            str(row["التصنيف الأساسي"]),
            str(row["التصنيف الفرعي"]),
            str(row["keywords"])
        ])

        normalized_text = normalize_text(
            searchable_text
        )

        text_tokens = set(
            normalized_text.split()
        )

        matched_terms = (
            query_word_set.intersection(
                text_tokens
            )
        )

        direct_score = (
            len(matched_terms)
            / len(query_word_set)
            if query_word_set
            else 0
        )

        direct_scores.append(
            direct_score
        )

        matched_terms_list.append(
            matched_terms
        )

    candidates["direct_match_score"] = (
        direct_scores
    )

    candidates["matched_terms"] = (
        matched_terms_list
    )

    # =====================================================
    # Final score
    # =====================================================

    candidates["final_score"] = (
        candidates["bge_score"] * 0.85
        +
        candidates["direct_match_score"] * 0.15
    )

    candidates = (
        candidates
        .sort_values(
            [
                "final_score",
                "bge_score"
            ],
            ascending=False
        )
        .reset_index(drop=True)
    )

    # =====================================================
    # Relevance Gate
    # =====================================================

    if candidates.empty:
        return pd.DataFrame()

    top_bge = candidates.iloc[0]["bge_score"]

    if top_bge < 0.001:
        return pd.DataFrame()

    # =====================================================
    # مستوى الصلة
    # =====================================================

    if top_bge >= 0.10:

        relevance_level = "strong"

    elif top_bge >= 0.005:

        relevance_level = "medium"

    else:

        relevance_level = "weak"

    # =====================================================
    # Threshold
    # =====================================================

    if relevance_level == "strong":

        threshold = max(
            0.01,
            top_bge * 0.02
        )

    elif relevance_level == "medium":

        threshold = max(
            0.002,
            top_bge * 0.50
        )

    else:

        threshold = max(
            0.001,
            top_bge * 0.50
        )

    results = candidates[
        candidates["bge_score"] >= threshold
    ].copy()

    # =====================================================
    # حماية إضافية
    # =====================================================

    if top_bge >= 0.50:

        strict_floor = max(
            0.01,
            top_bge * 0.05
        )

        results = results[
            (
                results["bge_score"]
                >= strict_floor
            )
            |
            (
                results["direct_match_score"]
                > 0
            )
        ].copy()

    if results.empty:
        return pd.DataFrame()

    # =====================================================
    # الأعمدة المطلوبة
    # =====================================================

    output_columns = [
        "ID",
        "اسم الدورة",
        "اسم المحاضرة",
        "التصنيف الأساسي",
        "التصنيف الفرعي",
        "keywords",
        "الرابط من المصدر الاساسي (حفظا لحقوق النشر)",
        "bge_score",
        "e5_score",
        "bm25_score",
        "direct_match_score",
        "final_score",
        "matched_terms"
    ]

    results = results[
        [
            col
            for col in output_columns
            if col in results.columns
        ]
    ]

    return results.reset_index(
        drop=True
    )


# =========================================================
# سبب ظهور النتيجة
# =========================================================

def build_result_reason(
    row,
    query
):

    query_normalized = normalize_text(
        query
    )

    query_words = set(
        query_normalized.split()
    )

    title = normalize_text(
        row.get("اسم المحاضرة", "")
    )

    category = normalize_text(
        row.get("التصنيف الأساسي", "")
    )

    subcategory = normalize_text(
        row.get("التصنيف الفرعي", "")
    )

    keywords = normalize_text(
        row.get("keywords", "")
    )

    # =====================================================
    # العنوان
    # =====================================================

    title_words = set(
        title.split()
    )

    title_matches = (
        query_words.intersection(
            title_words
        )
    )

    if len(title_matches) >= 2:

        return (
            "يتطابق موضوع البحث بشكل مباشر "
            "مع عنوان المحاضرة."
        )

    if len(title_matches) == 1:

        return (
            "يتضمن عنوان المحاضرة موضوعًا "
            "مرتبطًا مباشرة ببحثك."
        )

    # =====================================================
    # التصنيف الفرعي
    # =====================================================

    if subcategory:

        if any(
            word in subcategory
            for word in query_words
            if len(word) >= 3
        ):

            return (
                "المحاضرة تقع ضمن التصنيف الفرعي "
                "المرتبط بالبحث: "
                f"{row.get('التصنيف الفرعي', '')}."
            )

    # =====================================================
    # التصنيف الأساسي
    # =====================================================

    if category:

        if any(
            word in category
            for word in query_words
            if len(word) >= 3
        ):

            return (
                "المحاضرة تنتمي إلى المجال "
                "المرتبط بالبحث: "
                f"{row.get('التصنيف الأساسي', '')}."
            )

    # =====================================================
    # Keywords
    # =====================================================

    if keywords:

        if any(
            word in keywords
            for word in query_words
            if len(word) >= 3
        ):

            return (
                "تتضمن الكلمات المفتاحية للمحاضرة "
                "موضوعات مرتبطة ببحثك."
            )

    return (
        "المحاضرة مرتبطة دلاليًا "
        "بموضوع البحث."
    )


# =========================================================
# حساب درجة الصلة
# =========================================================

def calculate_relevance_score(results):

    if results.empty:
        return results

    results = results.copy()

    top_score = results["bge_score"].max()

    if top_score <= 0:

        results["relevance_score"] = 0

        return results

    results["relevance_score"] = (
        results["bge_score"]
        / top_score
        * 100
    )

    results["relevance_score"] = (
        results["relevance_score"]
        .clip(0, 100)
        .round(0)
        .astype(int)
    )

    return results


def display_results(results, query):
    if results.empty:
        st.warning(f"لم نجد محاضرات مناسبة للبحث: {query}")
        return

    results = calculate_relevance_score(results)

    # عدد النتائج - ناحية اليمين
    st.markdown(
        f"**تم العثور على {len(results)} نتيجة مناسبة**"
    )

    for _, row in results.iterrows():

        course = str(row["اسم الدورة"]).strip()
        lecture = str(row["اسم المحاضرة"]).strip()
        reason = build_result_reason(row, query)

        url = str(
            row["الرابط من المصدر الاساسي (حفظا لحقوق النشر)"]
        ).strip()

        score = int(row["relevance_score"])

        # =========================
        # بطاقة النتيجة
        # =========================
        with st.container(border=True):

            # تقسيم الكارد:
            # المحتوى يمين
            # درجة الصلة شمال
            col_content, col_score = st.columns([5, 1])

            # =========================
            # المحتوى - اليمين
            # =========================
            with col_content:

                st.markdown(
                    f"**الدورة:** {course}"
                )

                st.markdown(
                    f"### المحاضرة: {lecture}"
                )

                st.markdown(
                    "**لماذا ظهرت هذه المحاضرة؟**"
                )

                st.info(
                    reason,
                    icon="💡"
                )

                if url and url.lower() != "nan":
                    st.link_button(
                        "🔗 فتح المحاضرة",
                        url
                    )

            # =========================
            # درجة الصلة - الشمال
            # =========================
            with col_score:

                st.metric(
                    "درجة الصلة",
                    f"{score}/100"
                )
# =========================================================
# واجهة التطبيق
# =========================================================

st.markdown(
    """
    <div class="main-title">
        🔎 محرك البحث الذكي لفيديوهات أكاديمية الفلاح
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# البحث
# =========================================================

query = st.text_input(
    "ماذا تبحث؟",
    placeholder="مثال: أريد محاضرات عن الإسعافات الأولية"
)


search_button = st.button(
    "🔍 بحث",
    type="primary",
    use_container_width=True
)


# =========================================================
# تنفيذ البحث
# =========================================================

if search_button:

    if not query.strip():

        st.warning(
            "من فضلك اكتب سؤال البحث."
        )

    else:

        with st.spinner(
            "جاري البحث في محاضرات أكاديمية الفلاح"
        ):

            results = search_library(
                query
            )

        display_results(
            results,
            query
        )