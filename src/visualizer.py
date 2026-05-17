import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style='whitegrid', palette='muted')

class Visualizer:
    """Generates visualization charts for candidate ranking."""

    def __init__(self, output_dir: str = 'outputs/plots'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _save_plot(self, filename: str):
        path = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    def plot_candidate_scores(self, df: pd.DataFrame, job_role: str):
        if df.empty:
            return
        plt.figure(figsize=(10, 6))
        sns.barplot(x='composite_score', y='candidate_name', data=df, palette='viridis')
        plt.title(f'Candidate Scores for {job_role}')
        plt.xlabel('Composite Score (%)')
        plt.ylabel('Candidate')
        self._save_plot(f'candidate_scores_{job_role.replace(" ", "_")}.png')

    def plot_skill_frequency(self, skills, top_n: int = 15):
        if not skills:
            return
        counts = pd.Series(skills).value_counts().head(top_n)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=counts.values, y=counts.index, palette='coolwarm')
        plt.title('Top Skills Frequency')
        plt.xlabel('Count')
        plt.ylabel('Skill')
        self._save_plot('skill_frequency.png')

    def plot_tfidf_vs_skill(self, df: pd.DataFrame, job_role: str):
        if df.empty:
            return
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x='tfidf_score',
            y='skill_match_pct',
            hue='recommendation',
            data=df,
            palette='deep',
            s=100,
            edgecolor='w'
        )
        plt.title(f'TF-IDF vs Skill Match for {job_role}')
        plt.xlabel('TF-IDF Score (%)')
        plt.ylabel('Skill Match (%)')
        self._save_plot(f'tfidf_vs_skill_{job_role.replace(" ", "_")}.png')

    def plot_skill_gap_heatmap(self, gap_df: pd.DataFrame, job_role: str):
        if gap_df.empty:
            return
        plt.figure(figsize=(12, 6))
        sns.heatmap(gap_df, annot=True, cmap='YlGnBu', cbar=False, linewidths=0.5)
        plt.title(f'Skill Gap Heatmap for {job_role}')
        plt.xlabel('Required Skill')
        plt.ylabel('Candidate')
        self._save_plot(f'skill_gap_{job_role.replace(" ", "_")}.png')

    def plot_score_distribution(self, df: pd.DataFrame, job_role: str):
        if df.empty:
            return
        plt.figure(figsize=(10, 6))
        sns.histplot(df['composite_score'], kde=True, color='skyblue', bins=8)
        plt.title(f'Score Distribution for {job_role}')
        plt.xlabel('Composite Score (%)')
        plt.ylabel('Number of Candidates')
        self._save_plot(f'score_distribution_{job_role.replace(" ", "_")}.png')

    def plot_experience_vs_score(self, df: pd.DataFrame, job_role: str):
        if df.empty or 'experience' not in df.columns:
            return
        plt.figure(figsize=(10, 6))
        sns.regplot(x='experience', y='composite_score', data=df, scatter_kws={'s': 80})
        plt.title(f'Experience vs Composite Score for {job_role}')
        plt.xlabel('Years of Experience')
        plt.ylabel('Composite Score (%)')
        self._save_plot(f'experience_vs_score_{job_role.replace(" ", "_")}.png')

    def plot_confusion_matrix(self, y_true, y_pred):
        if len(y_true) == 0 or len(y_pred) == 0:
            return
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Rejected', 'Shortlisted'],
            yticklabels=['Rejected', 'Shortlisted'],
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        self._save_plot('confusion_matrix.png')
