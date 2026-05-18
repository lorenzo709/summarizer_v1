from deepeval.metrics import SummarizationMetric
from deepeval.test_case import LLMTestCase
from bert_score import BERTScorer
from rouge_score import rouge_scorer
from deepeval.models import OllamaModel
from deepeval import evaluate

from src.MyTypes import ResultPipeLine, EvaluationSingleSummary, EvaluationSummaries
import os
import glob
import json

input_folder = "./output_pipeline"
output_folder = "./result_evaluations"

os.makedirs(output_folder, exist_ok=True)

json_files = glob.glob(os.path.join(input_folder, "*.json"))
for file_path in json_files:
    file_name = os.path.basename(file_path)
    print(f"Processing: {file_name}") 

    try:
        with open(file_path, "r") as f:
            result_pipeline = ResultPipeLine.model_validate(json.read(f))

        # folder_path = Path("../knowledge")
        # inputs_papers = []
        # for pdf_file in folder_path.glob("*.pdf"):
        #     pdf_name = pdf_file.name
        #     pdf_path = str(pdf_file)
        #     parsed_text = parser(pdf_path)
        #     inputs_papers.append(parsed_text)
        
        # --- 2. DEEPEVAL (Alignment & Hallucination Check) ---
        # This uses an LLM to extract 'truths' from the source and compare to the summary.
        test_case = LLMTestCase(input=original_paper, actual_output=generated_summary)
        # n=5 generates 5 internal questions to check coverage
        summ_metric = SummarizationMetric(threshold=0.5, n=5) 

        # summ_metric.measure(test_case)

        result = evaluate(
            test_cases=[test_case],
            metrics=[summ_metric],#summarization_metric],
            print_results=True,
            run_async=False,
            verbose_mode=True
        )

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

    except Exception as e:
            print(f"Error processing {file_name}: {e}")
    # --- 1. SETUP DATA ---
    # original_paper = """
    # Your 5000+ word scientific paper text goes here. 
    # Let's assume it discusses a new CRISPR method for treating muscular dystrophy, 
    # noting a 15% increase in muscle fiber density in mice.
    # """

    # generated_summary = """
    # The paper introduces a CRISPR-based therapy for muscular dystrophy. 
    # The study achieved a 15% increase in muscle fiber density in murine models.
    # """

    # local_judge = OllamaModel(
    #     model="qwen3.5:9b",                  # Must match your Ollama model name exactly
    #     base_url="http://localhost:11434", # Default local Ollama endpoint
    # )

