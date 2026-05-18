from pathlib import Path
import sys

from deepeval.metrics import SummarizationMetric
from deepeval.test_case import LLMTestCase
from bert_score import BERTScorer
from rouge_score import rouge_scorer
from deepeval.models import OllamaModel
from deepeval import evaluate
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "..", "src")
tool_path = os.path.join(current_dir, "..", "tools")
sys.path.append(os.path.abspath(src_path))
sys.path.append(os.path.abspath(tool_path))

from tools.pdf_parser_no_tool_version import parser
from src.MyTypes import ResultPipeLine, EvaluationSingleSummary, EvaluationSummaries, PaperFound, ParsedText
import glob
import json

input_folder = "./output_pipeline"
output_folder = "./result_evaluations"

os.makedirs(output_folder, exist_ok=True)

def parsing_all_the_papers():

    knowledge_folders =["../knowledge", "../knowledge_platinum_water_splitting", "../knowledge_retrieval_augment_generation", "../knowledge_vision_transformers", "../knowledge_Zero_Shot_Robot_Manipulation"]
    parsed_papers = []

    for knowledge_folder in knowledge_folders:
        folder_path = Path(knowledge_folder)
        papers_to_parse = []
        for pdf_file in folder_path.glob("*.pdf"):
            pdf_name = pdf_file.name
            pdf_path = str(pdf_file)
            paper_found = PaperFound(pdf_name=pdf_name,pdf_path=pdf_path)
            papers_to_parse.append(paper_found)

        for paper in papers_to_parse:
            parsed_text = parser(paper.pdf_path)
            pdf_name = paper.pdf_name
            print(paper.pdf_path)
            print(paper.pdf_name)
            final_paper = ParsedText(pdf_name=pdf_name,parsed_text=parsed_text)
            parsed_papers.append(final_paper)

    return parsed_papers

def test_summary(original_paper, generated_summary):

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


json_files = glob.glob(os.path.join(input_folder, "*.json"))

parsed_papers = parsing_all_the_papers()
for file_path in json_files:
    file_name = os.path.basename(file_path)
    print(f"Processing: {file_name}") 

    try:
        with open(file_path, "r") as f:
            result_pipeline = ResultPipeLine.model_validate(json.read(f))

        for processed_paper in result_pipeline.processed_papers:
             original_paper = next(p for p in parsed_papers if p.pdf_name == processed_paper.paper_name)
             generated_summary = processed_paper.summary
             test_summary(original_paper, generated_summary)
        
    except Exception as e:
            print(f"Error processing {file_name}: {e}")

