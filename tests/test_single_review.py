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

import re

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

review_evaluation_metric = GEval(
    name="Scientific Review Quality",
    criteria="""
    Evaluate if the agent's review accurately defines the strong points (main benefits) 
    and limitations (areas for improvement) of the source scientific paper.
    
    CRITICAL FACTUALITY RULE: The review must not contain hallucinations. Every strong point 
    and limitation mentioned must be strictly supported by the text of the scientific paper. 
    If the review contains any false claims, fabricated data, or methods/results not found 
    in the source paper, you must penalize the score heavily.
    """,
    # G-Eval automatically maps these steps to a 0.0 - 1.0 mathematical score
    evaluation_steps=[
        "Read the provided source scientific paper and the agent's generated review.",
        "Cross-reference every claim in the 'Strong Points' section with the source paper. Ensure they are true and actual strengths.",
        "Cross-reference every claim in the 'Limitations' section. Ensure they are fair criticisms based strictly on the text.",
        "Check heavily for hallucinations: Did the agent invent any data, author names, results, or methodologies?",
        "Determine the final score between 0.0 and 1.0. If any hallucination is found, penalize the score heavily."
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.75 # The test will fail if the score drops below 0.75
)

def test_overview(original_paper, generated_review) -> float:

    test_case = LLMTestCase(input=original_paper, actual_output=generated_review)

    review_evaluation_metric.measure(test_case)
    print(f"--- DEEPEVAL RESULTS ---")
    print(f"Alignment Score: {review_evaluation_metric.score:.2f} (1.0 = Factually perfect)")
    print(f"Reasoning: {review_evaluation_metric.reason}\n")

    return review_evaluation_metric.score

input_folder = "./output_pipeline_corrected"
output_folder = "./result_evaluations_overview"


os.makedirs(output_folder, exist_ok=True)

json_files = glob.glob(os.path.join(input_folder, "*.json"))

parsed_papers = parsing_all_the_papers()

checkpoint_path = Path("checkpoint_single_review.json")

if checkpoint_path.is_file():
    with open("checkpoint_single_review.json", "r") as f:
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
                generated_overview = processed_paper.pros_and_cons
                single_review_eval= test_overview(original_paper, generated_overview)
                evaluation_result.evaluations.append(single_review_eval)

            filename = f"eval_{result_pipeline.topic}_{result_pipeline.model}.json"
            output_path = os.path.join(output_folder, filename)
            with open(output_path, "w") as f:
                f.write(evaluation_result.model_dump_json())

            results_evaluated.append(file_name)
            with open("checkpoint_single_review.json", "w") as f:
                f.write(json.dumps(results_evaluated))
        
    except Exception as e:
            print(f"Error processing {file_name}: {e}")