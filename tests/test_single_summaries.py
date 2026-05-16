from deepeval.metrics import SummarizationMetric
from deepeval.test_case import LLMTestCase
from bert_score import BERTScorer
from rouge_score import rouge_scorer
from deepeval.models import OllamaModel

# --- 1. SETUP DATA ---
original_paper = """
Your 5000+ word scientific paper text goes here. 
Let's assume it discusses a new CRISPR method for treating muscular dystrophy, 
noting a 15% increase in muscle fiber density in mice.
"""

generated_summary = """
The paper introduces a CRISPR-based therapy for muscular dystrophy. 
The study achieved a 15% increase in muscle fiber density in murine models.
"""

local_judge = OllamaModel(
    model="qwen3.5:9b",                  # Must match your Ollama model name exactly
    base_url="http://localhost:11434", # Default local Ollama endpoint
)

# --- 2. DEEPEVAL (Alignment & Hallucination Check) ---
# This uses an LLM to extract 'truths' from the source and compare to the summary.
test_case = LLMTestCase(input=original_paper, actual_output=generated_summary)
# n=5 generates 5 internal questions to check coverage
summ_metric = SummarizationMetric(threshold=0.5, n=5) 

summ_metric.measure(test_case)

print(f"--- DEEPEVAL RESULTS ---")
print(f"Alignment Score: {summ_metric.score:.2f} (1.0 = Factually perfect)")
print(f"Reasoning: {summ_metric.reason}\n")


# --- 3. BERTSCORE (Semantic Meaning) ---
# This ignores word counts and looks at the 'vibe' and technical meaning.
scorer = BERTScorer(lang="en", model_type="microsoft/deberta-xlarge-mnli")
P, R, F1 = scorer.score([generated_summary], [original_paper])

print(f"--- BERTSCORE RESULTS ---")
print(f"Semantic Similarity (F1): {F1.mean().item():.4f}\n")


# --- 4. ROUGE (Precision-focused) ---
# We focus on PRECISION here so we don't punish the brevity.
r_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
scores = r_scorer.score(original_paper, generated_summary)

print(f"--- ROUGE RESULTS ---")
# Precision: 'How much of my summary is actually found in the source?'
print(f"Groundedness (Precision): {scores['rougeL'].precision:.4f}")