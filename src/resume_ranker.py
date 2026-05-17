import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ResumeRanker:
    """Ranks candidates using TF-IDF similarity and skill matching."""

    def rank_candidates(
        self,
        df: pd.DataFrame,
        job_text: str,
        cleaned_resume_col: str,
        skill_match_col: str,
        tfidf_weight: float = 0.5,
        skill_weight: float = 0.5,
    ) -> pd.DataFrame:
        df_copy = df.copy()
        documents = df_copy[cleaned_resume_col].fillna('').astype(str).tolist()
        if len(documents) == 0:
            df_copy['tfidf_score'] = 0.0
            df_copy['composite_score'] = df_copy[skill_match_col].fillna(0.0)
            df_copy['recommendation'] = '🔴 Not Recommended'
            df_copy['rank'] = range(1, len(df_copy) + 1)
            return df_copy

        vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        corpus = documents + [job_text or '']
        tfidf_matrix = vectorizer.fit_transform(corpus)

        resume_vectors = tfidf_matrix[:-1]
        job_vector = tfidf_matrix[-1]

        similarity_scores = cosine_similarity(resume_vectors, job_vector).reshape(-1)
        df_copy['tfidf_score'] = (similarity_scores * 100).round(2)
        df_copy['composite_score'] = (
            df_copy['tfidf_score'] * tfidf_weight
            + df_copy[skill_match_col] * skill_weight
        ).round(2)

        def recommendation(score: float) -> str:
            if score >= 70:
                return '🟢 Highly Recommended'
            if score >= 50:
                return '🟡 Recommended'
            if score >= 30:
                return '🟠 Consider'
            return '🔴 Not Recommended'

        df_copy['recommendation'] = df_copy['composite_score'].apply(recommendation)
        df_copy = df_copy.sort_values(by='composite_score', ascending=False).reset_index(drop=True)
        df_copy['rank'] = df_copy.index + 1

        return df_copy

    def simulate_classification(self, df: pd.DataFrame, threshold: float):
        y_pred = (df['composite_score'] >= threshold).astype(int).to_numpy()
        y_true = (df['skill_match_pct'] >= threshold).astype(int).to_numpy()
        return y_true, y_pred
