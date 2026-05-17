import re
from typing import List, Set

SKILL_LOOKUP = [
    'python', 'sql', 'excel', 'power bi', 'powerbi', 'tableau', 'statistics',
    'pandas', 'numpy', 'machine learning', 'scikit-learn', 'scikitlearn',
    'tensorflow', 'keras', 'deep learning', 'nlp', 'natural language processing',
    'r', 'django', 'flask', 'rest api', 'api', 'postgresql', 'mysql', 'aws',
    'docker', 'git', 'html', 'css', 'javascript', 'react', 'node.js',
    'node js', 'mongodb', 'mongo db', 'fastapi', 'spacy', 'nltk', 'pytorch',
    'spark', 'hadoop', 'business intelligence', 'seaborn', 'matplotlib',
    'hugging face', 'google cloud', 'gcp', 'ci/cd', 'ui/ux', 'tensorflow',
    'pytorch', 'power bi', 'powerbi'
]

CANONICAL_SKILLS = {
    'powerbi': 'Power BI',
    'power bi': 'Power BI',
    'scikitlearn': 'Scikit-learn',
    'scikit-learn': 'Scikit-learn',
    'machine learning': 'Machine Learning',
    'deep learning': 'Deep Learning',
    'natural language processing': 'NLP',
    'rest api': 'REST API',
    'node.js': 'Node.js',
    'node js': 'Node.js',
    'mongo db': 'MongoDB',
    'aws': 'AWS',
    'google cloud': 'Google Cloud',
    'gcp': 'Google Cloud',
    'ci/cd': 'CI/CD',
    'ui/ux': 'UI/UX',
    'hugging face': 'Hugging Face',
}

for token in SKILL_LOOKUP:
    if token not in CANONICAL_SKILLS:
        CANONICAL_SKILLS[token] = token.title()

SKILL_PATTERNS = [
    (re.compile(r'\b' + re.escape(skill) + r'\b', re.IGNORECASE), skill)
    for skill in SKILL_LOOKUP
]

class SkillExtractor:
    """Extracts skills from resume and job description text."""

    def __init__(self):
        self.patterns = SKILL_PATTERNS
        self.canonical = CANONICAL_SKILLS

    def extract_skills(self, text: str) -> List[str]:
        if not isinstance(text, str):
            return []

        normalized = text.lower()
        found: Set[str] = set()

        for pattern, raw_skill in self.patterns:
            if pattern.search(normalized):
                found.add(self.canonical.get(raw_skill, raw_skill.title()))

        return sorted(found)

    def get_skill_match_percentage(self, candidate_skills: List[str], required_skills: List[str]) -> float:
        if not required_skills:
            return 0.0

        cand_set = {skill.lower() for skill in candidate_skills}
        req_set = {skill.lower() for skill in required_skills}
        matched = cand_set.intersection(req_set)
        return round(len(matched) / len(req_set) * 100.0, 2)

    def get_skill_gap(self, candidate_skills: List[str], required_skills: List[str]) -> List[str]:
        cand_set = {skill.lower() for skill in candidate_skills}
        return [skill for skill in required_skills if skill.lower() not in cand_set]
