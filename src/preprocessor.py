import re
from typing import List

STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has',
    'he', 'in', 'is', 'it', 'its', 'of', 'on', 'or', 'that', 'the', 'to',
    'was', 'were', 'will', 'with', 'this', 'such', 'their', 'they', 'them',
    'have', 'also', 'but', 'not', 'can', 'become', 'used', 'using', 'use',
    'into', 'more', 'other', 'these', 'those', 'which', 'about', 'should',
    'well', 'know', 'knows', 'new', 'years', 'year', 'work', 'works', 'worked'
}

class TextPreprocessor:
    """Simple text preprocessing for resume and job description data."""

    def __init__(self):
        self.stopwords = STOPWORDS

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ''

        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        tokens = [token for token in text.split() if token not in self.stopwords]
        return ' '.join(tokens)

    def preprocess_dataframe(self, df, text_col: str):
        if text_col not in df.columns:
            raise ValueError(f'Column {text_col} not found in DataFrame')
        return df[text_col].fillna('').astype(str).apply(self.clean_text)
