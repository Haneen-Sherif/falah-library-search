import os
import re
import html

import numpy as np
import pandas as pd
import gradio as gr

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi


# =========================================================
# إعدادات
# =========================================================

EXCEL_FILE = "Book.xlsx"

DESCRIPTION_COL = (
    "وصف المحاضرة \n"
    "او \n"
    "الدورة (في حال عدم وجود وصف للمحاضرات)"
)

# النماذج
E5_MODEL = "intfloat/multilingual-e5-base"
BGE_MODEL = "BAAI/bge-reranker-v2-m3"


# =========================================================
# 1) تحميل ملف Excel
# =========================================================

print("تحميل ملف Excel...")

df = pd.read_excel(EXCEL_FILE)

print("عدد الصفوف:", len(df))
print("عدد الأعمدة:", len(df.columns))


# =========================================================
# 2) تنظيف النص
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

    # توحيد بعض الحروف العربية
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
# 3) التأكد من الأعمدة المطلوبة
# =========================================================

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
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "الأعمدة التالية غير موجودة في ملف Excel:\n"
        + "\n".join(missing_columns)
    )


# =========================================================
# 4) تنظيف أعمدة البحث
# =========================================================

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


# =========================================================
# 5) بناء Search Profile
# =========================================================

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

print("تم إنشاء Search Profile.")


# =========================================================
# 6) تحميل Multilingual E5
# =========================================================

print("تحميل Multilingual E5...")

search_model = SentenceTransformer(
    E5_MODEL
)

print("تم تحميل E5.")


# =========================================================
# 7) إنشاء Embeddings للمحاضرات
# =========================================================

print("إنشاء Embeddings للمحاضرات...")

document_embeddings = search_model.encode(
    [
        "passage: " + text
        for text in df["search_profile"].tolist()
    ],
    normalize_embeddings=True,
    show_progress_bar=True
)

print(
    "عدد المحاضرات المفهرسة:",
    len(document_embeddings)
)


# =========================================================
# 8) تجهيز BM25
# =========================================================

print("إنشاء BM25...")

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


bm25 = BM25Okapi(
    bm25_corpus
)

print("تم إنشاء BM25.")


# =========================================================
# 9) تحميل BGE Reranker
# =========================================================

print("تحميل BGE Reranker...")

reranker = CrossEncoder(
    BGE_MODEL,
    max_length=512
)

print("تم تحميل BGE Reranker.")


# =========================================================
# 10) Hybrid Search
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


    # -----------------------------------------------------
    # BM25
    # -----------------------------------------------------

    bm25_scores = bm25.get_scores(
        query_tokens
    )

    bm25_order = np.argsort(
        -bm25_scores
    )


    # -----------------------------------------------------
    # E5
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # RRF
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # Candidate selection
    # -----------------------------------------------------

    candidates = df.copy()

    candidates["bm25_score"] = (
        bm25_scores
    )

    candidates["e5_score"] = (
        e5_scores
    )

    candidates["rrf_score"] = (
        rrf_scores
    )

    candidates = (
        candidates
        .sort_values(
            "rrf_score",
            ascending=False
        )
        .head(max_candidates)
        .reset_index(drop=True)
    )


    # -----------------------------------------------------
    # BGE Reranking
    # -----------------------------------------------------

    pairs = [
        [
            query,
            row["search_profile"]
        ]
        for _, row in candidates.iterrows()
    ]

    candidates["bge_score"] = (
        reranker.predict(
            pairs,
            show_progress_bar=False
        )
    )


    # -----------------------------------------------------
    # Direct topic matching
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # Final score
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # Relevance Gate
    # -----------------------------------------------------

    if candidates.empty:
        return pd.DataFrame()

    top_bge = candidates.iloc[0][
        "bge_score"
    ]


    # لا يوجد تطابق حقيقي
    if top_bge < 0.001:
        return pd.DataFrame()


    # -----------------------------------------------------
    # تحديد مستوى الصلة
    # -----------------------------------------------------

    if top_bge >= 0.10:

        relevance_level = "strong"

    elif top_bge >= 0.005:

        relevance_level = "medium"

    else:

        relevance_level = "weak"


    # -----------------------------------------------------
    # Threshold
    # -----------------------------------------------------

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
        candidates["bge_score"]
        >= threshold
    ].copy()


    # -----------------------------------------------------
    # حماية إضافية للموضوعات الواضحة
    # -----------------------------------------------------

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
                results[
                    "direct_match_score"
                ] > 0
            )
        ].copy()


    if results.empty:
        return pd.DataFrame()


    # -----------------------------------------------------
    # الأعمدة المطلوبة
    # -----------------------------------------------------

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
# 11) سبب ظهور النتيجة
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
        row.get(
            "اسم المحاضرة",
            ""
        )
    )

    category = normalize_text(
        row.get(
            "التصنيف الأساسي",
            ""
        )
    )

    subcategory = normalize_text(
        row.get(
            "التصنيف الفرعي",
            ""
        )
    )

    keywords = normalize_text(
        row.get(
            "keywords",
            ""
        )
    )


    # -----------------------------------------------------
    # العنوان
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # التصنيف الفرعي
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # التصنيف الأساسي
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # Keywords
    # -----------------------------------------------------

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
# 12) حساب درجة الصلة
# =========================================================

def calculate_relevance_score(
    results
):

    if results.empty:
        return results

    results = results.copy()

    top_score = results[
        "bge_score"
    ].max()

    if top_score <= 0:

        results[
            "relevance_score"
        ] = 0

        return results


    results[
        "relevance_score"
    ] = (
        results["bge_score"]
        / top_score
        * 100
    )

    results[
        "relevance_score"
    ] = (
        results[
            "relevance_score"
        ]
        .clip(0, 100)
        .round(0)
        .astype(int)
    )

    return results


# =========================================================
# 13) تجهيز النتائج
# =========================================================

def format_search_results(
    results,
    query
):

    if results.empty:
        return []

    results = calculate_relevance_score(
        results
    )

    formatted = []

    for _, row in results.iterrows():

        formatted.append({

            "id": row["ID"],

            "course": row[
                "اسم الدورة"
            ],

            "lecture": row[
                "اسم المحاضرة"
            ],

            "url": row[
                "الرابط من المصدر الاساسي (حفظا لحقوق النشر)"
            ],

            "relevance_score": int(
                row[
                    "relevance_score"
                ]
            ),

            "reason": build_result_reason(
                row,
                query
            )
        })

    return formatted


# =========================================================
# 14) Web Search
# =========================================================

def web_search(query):

    query = str(query).strip()

    if not query:

        return """
        <div style="
            direction:rtl;
            text-align:center;
            padding:30px;
        ">
            من فضلك اكتب سؤال البحث.
        </div>
        """


    results = search_library(
        query
    )

    formatted_results = (
        format_search_results(
            results,
            query
        )
    )


    # -----------------------------------------------------
    # لا توجد نتائج
    # -----------------------------------------------------

    if not formatted_results:

        safe_query = html.escape(
            query
        )

        return f"""
        <div style="
            direction:rtl;
            text-align:right;
            font-family:Arial;
            padding:30px;
            border-radius:18px;
            background:#f8f9fa;
            border:1px solid #e5e7eb;
        ">

            <h2 style="margin-top:0;">
                لم نجد محاضرات مناسبة
            </h2>

            <p style="color:#667085;">
                لا توجد محاضرات في المكتبة
                مرتبطة بشكل كافٍ ببحثك:
            </p>

            <strong>
                {safe_query}
            </strong>

        </div>
        """


    # -----------------------------------------------------
    # Cards
    # -----------------------------------------------------

    cards = []

    for result in formatted_results:

        course = html.escape(
            str(result["course"])
        )

        lecture = html.escape(
            str(result["lecture"])
        )

        reason = html.escape(
            str(result["reason"])
        )

        url = html.escape(
            str(result["url"])
        )

        score = int(
            result["relevance_score"]
        )


        card = f"""
        <div style="
            direction:rtl;
            text-align:right;
            font-family:Arial;
            background:white;
            border:1px solid #e5e7eb;
            border-radius:18px;
            padding:22px;
            margin:15px 0;
            box-shadow:0 5px 18px rgba(16,24,40,.06);
        ">

            <div style="
                color:#667085;
                font-size:14px;
                margin-bottom:8px;
            ">
                {course}
            </div>

            <div style="
                font-size:22px;
                font-weight:bold;
                color:#172033;
                margin-bottom:15px;
            ">
                {lecture}
            </div>

            <div style="
                background:#f8fafc;
                padding:13px;
                border-radius:12px;
                line-height:1.8;
                margin-bottom:17px;
                color:#344054;
            ">

                <strong>
                    لماذا ظهرت هذه المحاضرة؟
                </strong>

                <br>

                {reason}

            </div>

            <div style="
                display:flex;
                justify-content:space-between;
                align-items:center;
                gap:12px;
                flex-wrap:wrap;
            ">

                <div style="font-weight:bold;">

                    درجة الصلة:

                    <span style="
                        color:#2563eb;
                        font-size:20px;
                    ">
                        {score}/100
                    </span>

                </div>

                <a
                    href="{url}"
                    target="_blank"
                    style="
                        background:#2563eb;
                        color:white;
                        padding:10px 18px;
                        border-radius:10px;
                        text-decoration:none;
                        font-weight:bold;
                    "
                >
                    فتح المحاضرة
                </a>

            </div>

        </div>
        """

        cards.append(card)


    return f"""
    <div style="
        direction:rtl;
        text-align:right;
        font-family:Arial;
    ">

        <div style="
            color:#667085;
            margin-bottom:15px;
        ">

            تم العثور على
            <strong>
                {len(formatted_results)}
            </strong>
            نتيجة مناسبة

        </div>

        {''.join(cards)}

    </div>
    """


# =========================================================
# 15) Gradio Interface
# =========================================================

css = """
body {
    direction: rtl;
}

.gradio-container {
    direction: rtl;
}
"""


with gr.Blocks(
    title="محرك البحث الذكي للمحاضرات",
    css=css
) as demo:

    gr.Markdown(
        """
        # 🔎 محرك البحث الذكي لفيديوهات أكاديمية الفلاح

        اكتب ما تبحث عنه باللغة الطبيعية،
        وسيبحث النظام داخل مكتبة المحاضرات.
        """
    )


    with gr.Row():

        query_box = gr.Textbox(
            placeholder=(
                "مثال: أريد محاضرات عن "
                "الإسعافات الأولية"
            ),
            label="ماذا تبحث؟",
            scale=5
        )

        search_button = gr.Button(
            "🔍 بحث",
            variant="primary",
            scale=1
        )


    results_html = gr.HTML()


    search_button.click(
        fn=web_search,
        inputs=query_box,
        outputs=results_html
    )


    query_box.submit(
        fn=web_search,
        inputs=query_box,
        outputs=results_html
    )


# =========================================================
# 16) تشغيل Render
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            7860
        )
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )
