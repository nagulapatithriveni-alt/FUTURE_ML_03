# 🤖 FUTURE_ML_03 — Resume / Candidate Screening System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?style=for-the-badge&logo=scikit-learn)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)

**An end-to-end Machine Learning system that automatically screens and ranks resumes based on job descriptions using NLP techniques.**

</div>

---

## 📌 Project Overview

Manual resume screening is time-consuming and inconsistent. This project builds an **automated resume screening and ranking system** that:

- **Extracts technical skills** from unstructured resume text
- **Matches candidates** to job descriptions using TF-IDF + Cosine Similarity
- **Ranks candidates** by a composite match score
- **Identifies skill gaps** for each candidate
- **Visualizes results** through professional charts and heatmaps

---

## 🎯 Project Goals

| Goal | Status |
|------|--------|
| Resume text cleaning & preprocessing | ✅ Done |
| Skill extraction from raw text | ✅ Done |
| Job description matching | ✅ Done |
| Resume ranking system | ✅ Done |
| Skill gap identification | ✅ Done |
| NLP-based text analysis | ✅ Done |
| Model evaluation (classification metrics) | ✅ Done |
| Visualization graphs | ✅ Done |
| Professional project structure | ✅ Done |

---

## 🗂️ Project Structure

```
FUTURE_ML_03/
│
├── data/
│   ├── resumes.csv              # 10 candidate resume records
│   └── job_descriptions.csv     # 4 job role descriptions
│
├── notebooks/
│   └── Resume_Screening_System.ipynb  # Step-by-step Jupyter notebook
│
├── src/
│   ├── __init__.py
│   ├── preprocessor.py          # Text cleaning & preprocessing module
│   ├── skill_extractor.py       # Skill extraction using regex NER
│   ├── resume_ranker.py         # TF-IDF + Cosine Similarity ranking
│   └── visualizer.py            # All chart generation functions
│
├── models/                      # Saved model artifacts (future use)
│
├── outputs/
│   ├── ranked_candidates_*.csv  # Exported ranking results
│   └── plots/                   # All generated visualizations
│       ├── candidate_scores_*.png
│       ├── skill_frequency.png
│       ├── tfidf_vs_skill_*.png
│       ├── skill_gap_*.png
│       ├── score_distribution_*.png
│       ├── experience_vs_score_*.png
│       └── confusion_matrix.png
│
├── main.py                      # Complete pipeline runner
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🧠 Technical Approach

### 1. Text Preprocessing Pipeline
```
Raw Text → Lowercase → Remove URLs/Special Chars → Remove Stopwords
         → Tokenize → Simple Lemmatization → Clean Text
```

### 2. Skill Extraction
- Built a **custom skill dictionary** with 80+ technical skills
- Uses **regex pattern matching** to find skills in text
- Organized by category: Programming, ML Frameworks, Cloud, Databases, etc.

### 3. Resume Matching — TF-IDF + Cosine Similarity
```
TF-IDF Score:
  - Convert resume + job description to numerical vectors
  - TF = how often a word appears in this document
  - IDF = how rare the word is across all documents
  - High TF-IDF = word is important for this document

Cosine Similarity:
  - Measures the "angle" between two text vectors
  - Score 1.0 = perfect match | Score 0.0 = no match
```

### 4. Composite Scoring
```
Composite Score = (TF-IDF Score × 0.5) + (Skill Match % × 0.5)
```

### 5. Recommendation Tiers
| Score | Recommendation |
|-------|---------------|
| ≥ 70% | 🟢 Highly Recommended |
| 50–70% | 🟡 Recommended |
| 30–50% | 🟠 Consider |
| < 30%  | 🔴 Not Recommended |

---

## 📊 Datasets

### resumes.csv
| Column | Description |
|--------|-------------|
| `candidate_name` | Full name of the candidate |
| `resume_text` | Full resume in plain text |
| `skills` | Comma-separated skills list |
| `experience` | Years of experience |

**10 Candidates:** Aarav Shah, Priya Mehta, Rahul Verma, Sneha Patil, Arjun Nair, Divya Sharma, Karan Joshi, Neha Gupta, Vikram Singh, Ananya Iyer

### job_descriptions.csv
| Column | Description |
|--------|-------------|
| `job_role` | Title of the job |
| `required_skills` | Skills the job needs |
| `job_description` | Full job description |

**4 Job Roles:** Data Analyst, Machine Learning Intern, Python Developer, Web Developer

---

## 📈 Visualizations Generated

1. **Candidate Score Bar Chart** — Color-coded ranking with shortlist threshold
2. **Skill Frequency Chart** — Top 15 skills across all candidates
3. **TF-IDF vs Skill Match Scatter** — Quadrant analysis of candidate fit
4. **Skill Gap Heatmap** — ✓/✗ grid showing who has which required skills
5. **Score Distribution** — Histogram + box plot of all score types
6. **Experience vs Score** — Scatter plot with trend line
7. **Confusion Matrix** — Binary classification evaluation

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/FUTURE_ML_03.git
cd FUTURE_ML_03
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
# OR
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Pipeline
```bash
python main.py
```

### 5. Open the Notebook
```bash
jupyter notebook notebooks/Resume_Screening_System.ipynb
```

---

## ⚙️ Configuration

In `main.py`, change the `CONFIG` dictionary to customize:

```python
CONFIG = {
    "target_job":    "Data Analyst",   # Change job role here
    "tfidf_weight":  0.5,              # Weight for TF-IDF (0–1)
    "skill_weight":  0.5,              # Weight for skill match (0–1)
    "shortlist_threshold": 50,         # Minimum score to shortlist
}
```

**Available job roles:** `"Data Analyst"`, `"Machine Learning Intern"`, `"Python Developer"`, `"Web Developer"`

---

## 📦 Dependencies

```
pandas==2.1.0          # Data manipulation
numpy==1.24.3           # Numerical operations
scikit-learn==1.3.0    # TF-IDF, cosine similarity, metrics
matplotlib==3.7.2       # Plotting
seaborn==0.12.2         # Statistical visualization
jupyter==1.0.0          # Notebook environment
```

---

## 📏 Model Evaluation

The system is evaluated as a binary classification task:
- **Shortlisted** (Composite Score ≥ 50%) = Class 1
- **Rejected** (Composite Score < 50%) = Class 0

**Metrics Used:**
- Accuracy Score
- Classification Report (Precision, Recall, F1-Score)
- Confusion Matrix (TP, TN, FP, FN)

---

## 🔮 Future Improvements

| Improvement | Description |
|-------------|-------------|
| **Sentence-BERT** | Semantic matching — understands meaning, not just keywords |
| **spaCy NER** | More accurate, context-aware skill extraction |
| **Streamlit Dashboard** | Interactive web UI for non-technical recruiters |
| **PDF Resume Parser** | Parse real PDF resumes using pdfplumber |
| **Multi-role Scoring** | Rank one candidate across multiple job roles |
| **Active Learning** | Improve ranking with recruiter feedback |
| **LinkedIn Scraper** | Pull real job postings automatically |
| **Resume Builder Feedback** | Tell candidates exactly what to add to their resume |

---

## 🧑‍💻 Author

## Author

**Thriveni Nagulapati**  
Machine Learning Intern  

📧 nagulapatithriveni@gmail.com  
🔗 LinkedIn: https://www.linkedin.com/in/thriveni-nagulapati-838405285  
🐙 GitHub: https://github.com/nagulapatithriveni-alt                                                                                                     
 ## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

