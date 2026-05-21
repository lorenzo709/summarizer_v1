from pathlib import Path
import sys

from deepeval.metrics import SummarizationMetric, GEval 
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
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

from pdf_parser_no_tool_version import parser
from MyTypes import ResultPipeLine, EvaluationSingleSummary, EvaluationSummaries, PaperFound, ParsedText
import glob
import json


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


# 2. DEFINE THE G-EVAL CRITERIA AND STEPS
# G-Eval translates these criteria into custom scoring prompts natively.
scientific_alignment_metric = GEval(
    name="Scientific Hallucination and Alignment",
    criteria=(
        "Determine whether the generated summary is factually aligned with the original text. "
        "The summary must not introduce outside information, extrapolate trends, alter scientific "
        "data/metrics, or declare conclusions unsupported by the source text."
    ),
    evaluation_steps=[
        "Read through the original scientific text and identify all core facts, data points, and constraints.",
        "Examine the generated summary sentence by sentence.",
        "Cross-reference every metric, p-value, sample size, or technical claim in the summary against the original text.",
        "Flag an alignment error if the summary introduces concepts, results, or data points not found anywhere in the source text (Hallucination).",
        "Penalize heavily if the summary states a relationship or outcome as absolute when the paper states it as speculative or conditional.",
        "Provide a score from 0 to 1 based on factual correctness, where 1.0 means completely hallucination-free and aligned, and 0.0 means completely inaccurate."
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.85 # We demand high factual precision for science
)

bert_scorer = BERTScorer(lang="en", model_type="microsoft/deberta-xlarge-mnli")
r_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def test_summary(original_paper, generated_summary) -> EvaluationSingleSummary:

    # --- 2. DEEPEVAL (Alignment & Hallucination Check) ---
    # This uses an LLM to extract 'truths' from the source and compare to the summary.
    test_case = LLMTestCase(input=original_paper, actual_output=generated_summary)
    # n=5 generates 5 internal questions to check coverage
    # summ_metric = SummarizationMetric(threshold=0.5, n=5) 

    # summ_metric.measure(test_case)

    scientific_alignment_metric.measure(test_case)
    print(f"--- DEEPEVAL RESULTS ---")
    print(f"Alignment Score: {scientific_alignment_metric.score:.2f} (1.0 = Factually perfect)")
    print(f"Reasoning: {scientific_alignment_metric.reason}\n")

    # --- 3. BERTSCORE (Semantic Meaning) ---
    # This ignores word counts and looks at the 'vibe' and technical meaning.
    P, R, F1 = bert_scorer.score([generated_summary], [original_paper])

    print(f"--- BERTSCORE RESULTS ---")
    print(f"Semantic Similarity (F1): {F1.mean().item():.4f}\n")

    # --- 4. ROUGE (Precision-focused) ---
    # We focus on PRECISION here so we don't punish the brevity.
    scores = r_scorer.score(original_paper, generated_summary)

    print(f"--- ROUGE RESULTS ---")
    # Precision: 'How much of my summary is actually found in the source?'
    print(f"Groundedness (Precision): {scores['rougeL'].precision:.4f}")

    single_sum_eval = EvaluationSingleSummary (
        deepeval = scientific_alignment_metric.score,
        BERT_score = F1.mean().item(),
        Rouge_L = scores['rougeL'].precision
    )

    return single_sum_eval


input_folder = "./output_pipeline"
output_folder = "./result_evaluations"

os.makedirs(output_folder, exist_ok=True)

json_files = glob.glob(os.path.join(input_folder, "*.json"))

parsed_papers = parsing_all_the_papers()

checkpoint_path = Path("checkpoint.json")

if checkpoint_path.is_file():
    with open("checkpoint.json", "r") as f:
        results_evaluated = json.load(f)
else:
    results_evaluated = []

for file_path in json_files:
    file_name = os.path.basename(file_path)

    if file_name in results_evaluated:
        print(f"Already evaluated: {file_name}")
        continue

    print(f"Processing: {file_name}") 

    try:
        with open(file_path, "r") as f:
            result_pipeline = ResultPipeLine.model_validate_json(f.read())

            evaluation_result = EvaluationSummaries(
                topic = result_pipeline.topic,
                model = result_pipeline.model,
                evaluations = []
            )

            for processed_paper in result_pipeline.processed_papers:
                raw_paper = next(p for p in parsed_papers if p.pdf_name == processed_paper.paper_name)
                print(raw_paper.pdf_name)
                print(processed_paper.paper_name)

                if raw_paper.pdf_name == processed_paper.paper_name:
                    print("Paper name match")

                original_paper = raw_paper.parsed_text
                generated_summary = processed_paper.summary
                single_summary_eval = test_summary(original_paper, generated_summary)
                evaluation_result.evaluations.append(single_summary_eval)

            filename = f"eval_{result_pipeline.topic}_{result_pipeline.model}.json"
            output_path = os.path.join(output_folder, filename)
            with open(output_path, "w") as f:
                f.write(evaluation_result.model_dump_json())

            results_evaluated.append(file_name)
            with open("checkpoint.json", "w") as f:
                f.write(json.dumps(results_evaluated))
        
    except Exception as e:
            print(f"Error processing {file_name}: {e}")

