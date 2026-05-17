"""
============================================================
  FUTURE_ML_03 - Resume / Candidate Screening System
  File: main.py
  Author: [Your Name]
  
  This is the MAIN file that runs the entire pipeline.
  Think of it like the "director" — it calls all the other
  modules in the right order to produce the final results.
  
  Pipeline Steps:
    1. Load Data
    2. Preprocess Text
    3. Extract Skills
    4. Match & Rank Candidates
    5. Identify Skill Gaps
    6. Evaluate Model
    7. Generate Visualizations
    8. Export Results
============================================================
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── Add src to path so we can import our modules ───────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# ── Import our custom modules ──────────────────────────────────────────────
from preprocessor    import TextPreprocessor
from skill_extractor import SkillExtractor
from resume_ranker   import ResumeRanker
from visualizer      import Visualizer

# ── Scikit-learn for model evaluation ─────────────────────────────────────
from sklearn.metrics import (classification_report, accuracy_score,
                              confusion_matrix)


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    "data_dir":      "data",
    "output_dir":    "outputs",
    "plots_dir":     "outputs/plots",
    "resume_file":   "data/resumes.csv",
    "job_file":      "data/job_descriptions.csv",
    "target_job":    "Data Analyst",   # ← Change this to test different roles
    "tfidf_weight":  0.5,              # Weight for TF-IDF score
    "skill_weight":  0.5,              # Weight for skill match score
    "shortlist_threshold": 50,         # Score cutoff for shortlisting
}


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def print_section(title):
    """Print a nicely formatted section header."""
    width = 65
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")

def print_step(step_num, description):
    """Print a step indicator."""
    print(f"\n[Step {step_num}] {description}")
    print("-" * 50)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1: LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    """
    Load resume and job description datasets from CSV files.
    
    Returns:
        tuple: (resume_df, job_df) — two DataFrames
    """
    print_step(1, "Loading Data")
    
    # Load resumes
    resume_df = pd.read_csv(CONFIG["resume_file"])
    print(f"  ✓ Loaded {len(resume_df)} resumes")
    print(f"    Columns: {list(resume_df.columns)}")
    
    # Load job descriptions
    job_df = pd.read_csv(CONFIG["job_file"])
    print(f"  ✓ Loaded {len(job_df)} job descriptions")
    print(f"    Job Roles: {list(job_df['job_role'])}")
    
    # Basic data validation
    assert 'resume_text' in resume_df.columns, "resume_text column missing!"
    assert 'job_role'    in job_df.columns,    "job_role column missing!"
    
    return resume_df, job_df


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2: TEXT PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_text(resume_df, job_df):
    """
    Clean and preprocess all text data.
    
    Parameters:
        resume_df (pd.DataFrame): Raw resume data
        job_df    (pd.DataFrame): Raw job description data
        
    Returns:
        tuple: (resume_df, job_df) with new 'cleaned_text' columns
    """
    print_step(2, "Text Preprocessing")
    
    preprocessor = TextPreprocessor()
    
    print("  Processing resumes...")
    resume_df['cleaned_text'] = preprocessor.preprocess_dataframe(
        resume_df, 'resume_text'
    )
    
    print("  Processing job descriptions...")
    job_df['cleaned_text'] = preprocessor.preprocess_dataframe(
        job_df, 'job_description'
    )
    
    # Show a sample cleaned text
    print(f"\n  Sample cleaned resume (first 100 chars):")
    print(f"  '{resume_df['cleaned_text'].iloc[0][:100]}...'")
    
    return resume_df, job_df


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3: SKILL EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_skills(resume_df, job_df):
    """
    Extract skills from resumes and job descriptions.
    
    Parameters:
        resume_df (pd.DataFrame): Preprocessed resume data
        job_df    (pd.DataFrame): Preprocessed job data
        
    Returns:
        tuple: (resume_df, job_df) with 'extracted_skills' columns
    """
    print_step(3, "Skill Extraction")
    
    extractor = SkillExtractor()
    
    # Extract skills from resume text
    print("  Extracting skills from resumes...")
    resume_df['extracted_skills'] = resume_df['resume_text'].apply(
        extractor.extract_skills
    )
    
    # Extract skills from job descriptions
    print("  Extracting skills from job descriptions...")
    job_df['extracted_skills'] = job_df['job_description'].apply(
        extractor.extract_skills
    )
    
    # Show extracted skills for each candidate
    print("\n  Extracted Skills Summary:")
    for _, row in resume_df.iterrows():
        skills_str = ', '.join(row['extracted_skills'][:5])  # Show first 5
        more = len(row['extracted_skills']) - 5
        if more > 0:
            skills_str += f" (+{more} more)"
        print(f"    {row['candidate_name']:15s} → {skills_str}")
    
    return resume_df, job_df, extractor


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4: JOB MATCHING & RANKING
# ══════════════════════════════════════════════════════════════════════════════

def match_and_rank(resume_df, job_df, extractor, target_job):
    """
    Match resumes against the target job and rank candidates.
    
    Parameters:
        resume_df  (pd.DataFrame): Resume data with extracted skills
        job_df     (pd.DataFrame): Job data with extracted skills
        extractor  (SkillExtractor): Skill extractor instance
        target_job (str):          The job role to screen for
        
    Returns:
        tuple: (ranked_df, job_row, ranker)
    """
    print_step(4, f"Matching & Ranking for: '{target_job}'")
    
    # Get the specific job row
    job_row = job_df[job_df['job_role'] == target_job].iloc[0]
    required_skills = job_row['extracted_skills']
    
    print(f"  Required Skills for '{target_job}':")
    print(f"  {', '.join(required_skills)}")
    
    # Calculate skill match percentage for each candidate
    print("\n  Calculating skill match percentages...")
    resume_df['skill_match_pct'] = resume_df['extracted_skills'].apply(
        lambda candidate_skills: extractor.get_skill_match_percentage(
            candidate_skills, required_skills
        )
    )
    
    # Rank candidates using TF-IDF + Cosine Similarity
    print("\n  Running TF-IDF ranking...")
    ranker = ResumeRanker()
    
    ranked_df = ranker.rank_candidates(
        df=resume_df,
        job_text=job_row['cleaned_text'],
        cleaned_resume_col='cleaned_text',
        skill_match_col='skill_match_pct',
        tfidf_weight=CONFIG['tfidf_weight'],
        skill_weight=CONFIG['skill_weight']
    )
    
    return ranked_df, job_row, ranker


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5: SKILL GAP ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_skill_gaps(ranked_df, job_row, extractor):
    """
    Identify skill gaps for each candidate.
    
    Parameters:
        ranked_df (pd.DataFrame): Ranked candidates
        job_row   (pd.Series):    Job description data
        extractor (SkillExtractor): Skill extractor instance
        
    Returns:
        pd.DataFrame: DataFrame with skill gap column added
    """
    print_step(5, "Skill Gap Analysis")
    
    required_skills = job_row['extracted_skills']
    
    # Calculate missing skills for each candidate
    ranked_df['skill_gap'] = ranked_df['extracted_skills'].apply(
        lambda c_skills: extractor.get_skill_gap(c_skills, required_skills)
    )
    
    # Display skill gaps
    print(f"\n  Skill Gaps (Missing Skills) for '{job_row['job_role']}':\n")
    for _, row in ranked_df.iterrows():
        status = "🟢" if row['composite_score'] >= 70 else \
                 "🟡" if row['composite_score'] >= 50 else \
                 "🟠" if row['composite_score'] >= 30 else "🔴"
        
        gap_str = ', '.join(row['skill_gap']) if row['skill_gap'] else "None! ✨"
        print(f"  {status} {row['candidate_name']:15s} (Score: {row['composite_score']:5.1f}%)")
        print(f"     Missing: {gap_str}\n")
    
    return ranked_df


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6: MODEL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(ranked_df, ranker):
    """
    Evaluate the screening system using classification metrics.
    
    We treat it as a binary classification problem:
      - Shortlisted (score >= threshold) = Class 1
      - Rejected    (score < threshold)  = Class 0
    
    Parameters:
        ranked_df (pd.DataFrame): Ranked candidates
        ranker    (ResumeRanker): Ranker with simulate_classification
        
    Returns:
        tuple: (y_true, y_pred)
    """
    print_step(6, "Model Evaluation")
    
    threshold = CONFIG['shortlist_threshold']
    y_true, y_pred = ranker.simulate_classification(ranked_df, threshold)
    
    accuracy = accuracy_score(y_true, y_pred)
    
    print(f"  Shortlist Threshold: {threshold}%")
    print(f"  Total Candidates:    {len(ranked_df)}")
    print(f"  Shortlisted:         {sum(y_pred == 1)}")
    print(f"  Rejected:            {sum(y_pred == 0)}")
    print(f"\n  Accuracy Score:      {accuracy:.2%}")
    
    print("\n  Classification Report:")
    print("-" * 50)
    report = classification_report(
        y_true, y_pred,
        target_names=['Rejected', 'Shortlisted'],
        zero_division=0
    )
    print(report)
    
    # Print confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("  Confusion Matrix:")
    print(f"  {'':15} Pred: Rejected  Pred: Shortlisted")
    print(f"  {'True: Rejected':15}      {cm[0][0]:4d}            {cm[0][1]:4d}")
    print(f"  {'True: Shortlisted':15}      {cm[1][0]:4d}            {cm[1][1]:4d}")
    
    return y_true, y_pred


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 7: VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════

def create_visualizations(ranked_df, resume_df, job_row, y_true, y_pred):
    """
    Generate all project visualizations and save them to disk.
    
    Parameters:
        ranked_df  (pd.DataFrame): Ranked candidates
        resume_df  (pd.DataFrame): Full resume dataset
        job_row    (pd.Series):    Job description data
        y_true     (array):        True labels
        y_pred     (array):        Predicted labels
    """
    print_step(7, "Creating Visualizations")
    
    viz = Visualizer(output_dir=CONFIG["plots_dir"])
    job_role = job_row['job_role']
    
    # Plot 1: Candidate Scores Bar Chart
    print("  Plotting candidate scores...")
    viz.plot_candidate_scores(ranked_df, job_role)
    
    # Plot 2: Skill Frequency Chart
    print("  Plotting skill frequency...")
    all_skills = []
    for skills in resume_df['extracted_skills']:
        all_skills.extend(skills)
    viz.plot_skill_frequency(all_skills, top_n=15)
    
    # Plot 3: TF-IDF vs Skill Match Scatter Plot
    print("  Plotting TF-IDF vs skill match...")
    viz.plot_tfidf_vs_skill(ranked_df, job_role)
    
    # Plot 4: Skill Gap Heatmap
    print("  Plotting skill gap heatmap...")
    required_skills = job_row['extracted_skills'][:8]  # Show top 8 skills
    
    # Build a binary matrix: candidate x skill (1=has, 0=missing)
    gap_data = {}
    for _, row in ranked_df.iterrows():
        name = row['candidate_name'].split()[0]  # First name
        gap_data[name] = {
            skill: (1 if skill in row['extracted_skills'] else 0)
            for skill in required_skills
        }
    
    gap_df = pd.DataFrame(gap_data).T
    if not gap_df.empty:
        viz.plot_skill_gap_heatmap(gap_df, job_role)
    
    # Plot 5: Score Distribution
    print("  Plotting score distribution...")
    viz.plot_score_distribution(ranked_df, job_role)
    
    # Plot 6: Experience vs Score
    print("  Plotting experience vs score...")
    viz.plot_experience_vs_score(ranked_df, job_role)
    
    # Plot 7: Confusion Matrix
    print("  Plotting confusion matrix...")
    viz.plot_confusion_matrix(y_true, y_pred)
    
    print(f"\n  ✓ All visualizations saved to: {CONFIG['plots_dir']}/")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 8: EXPORT RESULTS
# ══════════════════════════════════════════════════════════════════════════════

def export_results(ranked_df, job_role):
    """
    Export final results to CSV files for reporting.
    
    Parameters:
        ranked_df (pd.DataFrame): Ranked candidates
        job_role  (str):          Job role name
    """
    print_step(8, "Exporting Results")
    
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    # Prepare export DataFrame
    export_cols = [
        'rank', 'candidate_name', 'experience',
        'skill_match_pct', 'tfidf_score', 'composite_score',
        'recommendation', 'skill_gap'
    ]
    
    # Convert skill_gap list to string for CSV
    export_df = ranked_df[export_cols].copy()
    export_df['skill_gap'] = export_df['skill_gap'].apply(
        lambda x: ', '.join(x) if x else 'None'
    )
    
    # Save to CSV
    output_path = os.path.join(
        CONFIG["output_dir"],
        f"ranked_candidates_{job_role.replace(' ', '_')}.csv"
    )
    export_df.to_csv(output_path, index=False)
    print(f"  ✓ Results saved to: {output_path}")
    
    # Print final ranking table
    print(f"\n  📋 FINAL RANKING — {job_role}")
    print("  " + "=" * 80)
    print(f"  {'Rank':4} {'Candidate':15} {'Exp':4} {'Skill%':8} {'TFIDF%':8} {'Score':8} {'Status'}")
    print("  " + "-" * 80)
    
    for _, row in export_df.iterrows():
        print(f"  #{row['rank']:<3} {row['candidate_name']:15} "
              f"{row['experience']:4}y "
              f"{row['skill_match_pct']:7.1f}% "
              f"{row['tfidf_score']:7.1f}% "
              f"{row['composite_score']:7.1f}% "
              f"{row['recommendation']}")
    
    print("  " + "=" * 80)
    
    return export_df


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Run the complete Resume Screening Pipeline.
    This is the entry point of the entire system.
    """
    print_section("FUTURE_ML_03 — RESUME / CANDIDATE SCREENING SYSTEM")
    print(f"  Target Job Role: {CONFIG['target_job']}")
    print(f"  TF-IDF Weight:   {CONFIG['tfidf_weight'] * 100}%")
    print(f"  Skill Weight:    {CONFIG['skill_weight'] * 100}%")
    print(f"  Shortlist At:    {CONFIG['shortlist_threshold']}%")
    
    # Create output directories
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(CONFIG["plots_dir"],  exist_ok=True)
    
    # ── Run All Pipeline Steps ─────────────────────────────────────────────
    
    # Step 1: Load
    resume_df, job_df = load_data()
    
    # Step 2: Preprocess
    resume_df, job_df = preprocess_text(resume_df, job_df)
    
    # Step 3: Extract Skills
    resume_df, job_df, extractor = extract_skills(resume_df, job_df)
    
    # Step 4: Match & Rank
    ranked_df, job_row, ranker = match_and_rank(
        resume_df, job_df, extractor, CONFIG['target_job']
    )
    
    # Step 5: Skill Gap
    ranked_df = analyze_skill_gaps(ranked_df, job_row, extractor)
    
    # Step 6: Evaluate
    y_true, y_pred = evaluate_model(ranked_df, ranker)
    
    # Step 7: Visualize
    create_visualizations(ranked_df, resume_df, job_row, y_true, y_pred)
    
    # Step 8: Export
    export_df = export_results(ranked_df, CONFIG['target_job'])
    
    # ── Done! ──────────────────────────────────────────────────────────────
    print_section("PIPELINE COMPLETE ✓")
    print(f"  Results saved to:       {CONFIG['output_dir']}/")
    print(f"  Visualizations saved to:{CONFIG['plots_dir']}/")
    print(f"\n  🏆 Top Candidate: {ranked_df.iloc[0]['candidate_name']}")
    print(f"     Score: {ranked_df.iloc[0]['composite_score']}%")
    print(f"     Status: {ranked_df.iloc[0]['recommendation']}")
    print()


if __name__ == "__main__":
    main()
