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
# 1. Define a metric that judges impact based ONLY on the topic
# topic_impact_metric = GEval(
#     name="Topic-Based Research Innovation",
#     criteria="""
#     Determine if the 'actual_output' provides innovative research opportunities 
#     specifically for the given 'input' (the topic). 
#     - The suggestions should be non-obvious.
#     - They should address current trends in the field of the topic.
#     - Penalize generic answers like 'more data is needed'.
#     """,
#     evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
#     threshold=0.7
# )

research_gap_metric = GEval(
    name="Research Gap Report Quality",
    criteria="""
    Evaluate the quality of a continuous text report analyzing aggregated research limitations.
    
    The agent must:
    1. Systematically categorize gaps in Methodology, Data, Scope, or Theory based ONLY on the input limitations.
    2. Ensure each opportunity is a clear research question/proposal, is directly justified by the input, and focuses on filling a SOTA hole.
    
    CRITICAL FACTUALITY & ALIGNMENT RULE: If the agent suggests research opportunities that are 
    NOT justified by the provided input text (i.e., external hallucinations or generic ideas), 
    heavily penalize the score.
    """,
    # or if it fails to include the exact title **'High-Impact Research Opportunities'**, 
    # 2. Include a bulleted list titled exactly **'High-Impact Research Opportunities'**.
    evaluation_steps=[
        "Verify that the output is a structured, continuous text report categorizing gaps (Methodology, Data, Scope, or Theory).",
        # "Check for the exact bolded header: **'High-Impact Research Opportunities'** followed by a bulleted list.",
        "Evaluate each research opportunity in the list: Is it clearly articulated as a research question or proposal?",
        "Cross-reference each opportunity with the input text: Is it explicitly justified by the limitations provided, or did the agent hallucinate/invent external problems?",
        "Assess if the opportunities genuinely address filling a major 'hole' in the current State-of-the-Art based on the text.",
        "Assign a score from 0.0 to 1.0. Deduct heavily for any unjustified opportunities or severe structural omissions."
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.80  # Fails the test if the score is below 0.80
)

synthesis_quality_metric = GEval(
    name="Cross-Document Synthesis Integrity",
    criteria=(
        "Evaluate how effectively the final summary aggregates multiple research papers. "
        "The goal is to move beyond a simple list and create a cohesive thematic narrative "
        "that identifies commonalities and contradictions across the sources."
    ),
    evaluation_steps=[
        "1. Compare the final summary against the input list of paper summaries.",
        "2. Check for 'Thematic Integration': Does it group similar findings together, or just list papers one by one?",
        "3. Verify 'Conflict Resolution': If papers have different results, does the summary acknowledge the disagreement?",
        "4. Check for 'Source Attribution': Can you still tell which findings came from which paper (or group of papers)?",
        "5. HEAVILY PENALIZE the score if any core methodology from a source paper is omitted or misrepresented.",
        "6. Assess if the summary provides a higher-level 'meta-analysis' rather than just repeating the input text."
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.8
)

def run_geval_test(metric: GEval, original_paper: str, generated_review: str) -> float:

    test_case = LLMTestCase(input=original_paper, actual_output=generated_review)

    metric.measure(test_case)
    print(f"--- DEEPEVAL RESULTS ---")
    print(f"Alignment Score: {metric.score:.2f} (1.0 = Factually perfect)")
    print(f"Reasoning: {metric.reason}\n")

    return metric.score


input_folder = "./output_pipeline_corrected"
output_folder = "./result_evaluations_SOTA_f_summary"


os.makedirs(output_folder, exist_ok=True)

json_files = glob.glob(os.path.join(input_folder, "*.json"))

parsed_papers = parsing_all_the_papers()

checkpoint_path = Path("checkpoint_SOTA_f_sum.json")

if checkpoint_path.is_file():
    with open("checkpoint_SOTA_f_sum.json", "r") as f:
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

            scores = {
                "SOTA" : float,
                "final_summary": float
            }

            # raw papers version
            paper_map = {p.pdf_name: p for p in parsed_papers}

            all_raw_papers = "\n".join(paper_map[proc.paper_name].parsed_text for proc in result_pipeline.processed_papers)

            # raw_papers = []
            # for processed_paper in result_pipeline.processed_papers:
            #     raw_paper = next(p for p in parsed_papers if p.pdf_name == processed_paper.paper_name)
            #     raw_papers.append(raw_paper)

            # all_raw_papers = "\n".join(raw_papers)
            
            # summaries/reviews version
            # all_summs = [p.summary for p in result_pipeline.processed_papers]
            # all_reviews= [p.pros_and_cons for p in result_pipeline.processed_papers]

            SOTA_generated = result_pipeline.gaps_in_SOTA
            final_summary_generated = result_pipeline.final_summary

            scores["SOTA"] = run_geval_test(research_gap_metric,all_raw_papers, SOTA_generated)
            scores["final_summary"] = run_geval_test(synthesis_quality_metric,all_raw_papers, final_summary_generated)
            evaluation_result.evaluations.append(scores)

            # for processed_paper in result_pipeline.processed_papers:
            #     raw_paper = next(p for p in parsed_papers if p.pdf_name == processed_paper.paper_name)
            #     print(raw_paper.pdf_name)
            #     print(processed_paper.paper_name)

            #     if raw_paper.pdf_name == processed_paper.paper_name:
            #         print("Paper name match")

            #     original_paper = raw_paper.parsed_text
            #     generated_overview = processed_paper.pros_and_cons
            #     single_review_eval= test_overview(original_paper, generated_overview)
            #     evaluation_result.evaluations.append(single_review_eval)

            filename = f"eval_{result_pipeline.topic}_{result_pipeline.model}.json"
            output_path = os.path.join(output_folder, filename)
            with open(output_path, "w") as f:
                f.write(evaluation_result.model_dump_json())

            results_evaluated.append(file_name)
            with open("checkpoint_SOTA_f_sum.json", "w") as f:
                f.write(json.dumps(results_evaluated))
        
    except Exception as e:
            print(f"Error processing {file_name}: {e}")