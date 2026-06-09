from pathlib import Path
import sys

from deepeval.metrics import SummarizationMetric, GEval 
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from bert_score import BERTScorer
from rouge_score import rouge_scorer
from deepeval.models import OllamaModel
from deepeval import evaluate
import os
from pydantic import ValidationError

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "..", "src")
tool_path = os.path.join(current_dir, "..", "tools")
sys.path.append(os.path.abspath(src_path))
sys.path.append(os.path.abspath(tool_path))

from pdf_parser_no_tool_version import parser
from MyTypes import ResultPipeLine, EvaluationSingleSummary, EvaluationSummaries, PaperFound, ParsedText, SummaryProConsSinglePaper
import glob
import json

import re

def parsing_all_the_papers():

    knowledge_folders =["../knowledge_feynman/liquid-nn-cst", "../knowledge_feynman/platinum-water-splitting", "../knowledge_feynman/rag-legacy-refactor", "../knowledge_feynman/vision-transformers", "../knowledge_feynman/zero-shot-clip-manipulation"]

    list_paper_parsed_by_topic = []

    for knowledge_folder in knowledge_folders:
        papers_by_topic = { }
        folder_path = Path(knowledge_folder)
        parsed_papers = []
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

        papers_by_topic["topic"] = knowledge_folder.split("/")[-1]
        print(papers_by_topic["topic"])
        papers_by_topic["parsed_papers"] = parsed_papers
        list_paper_parsed_by_topic.append(papers_by_topic)

    return list_paper_parsed_by_topic

research_gap_metric = GEval(
    name="Research Gap Report Quality",
    criteria="""
    Evaluate the quality of a continuous text (or list) report analyzing aggregated research limitations.
    
    The agent must:
    1. Systematically categorize gaps based ONLY on the input limitations.
    2. Ensure each opportunity is a clear research question/proposal, is directly justified by the input, and focuses on filling a SOTA hole.
    
    CRITICAL FACTUALITY & ALIGNMENT RULE: If the agent suggests research opportunities that are 
    NOT justified by the provided input text (i.e., external hallucinations or generic ideas), 
    heavily penalize the score.
    """,
    # or if it fails to include the exact title **'High-Impact Research Opportunities'**, 
    # 2. Include a bulleted list titled exactly **'High-Impact Research Opportunities'**.
    evaluation_steps=[
        "Verify that the output is a structured, continuous text (or list) report categorizing gaps.",
        # "Check for the exact bolded header: **'High-Impact Research Opportunities'** followed by a bulleted list.",
        "Evaluate each research opportunity in the list: Is it clearly articulated as a research question or proposal?",
        "Cross-reference each opportunity with the input text: Is it explicitly justified by the limitations provided, or did the agent hallucinate/invent external problems?",
        "Assess if the opportunities genuinely address filling a major 'hole' in the current State-of-the-Art based on the text.",
        "Assign a score from 0.0 to 1.0. Deduct heavily for any unjustified opportunities or severe structural omissions."
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.80  # Fails the test if the score is below 0.80
)

def run_geval_test(metric: GEval, original_paper: str, generated_review: str) -> float:

    test_case = LLMTestCase(input=original_paper, actual_output=generated_review)

    metric.measure(test_case)
    print(f"--- DEEPEVAL RESULTS ---")
    print(f"Alignment Score: {metric.score:.2f} (1.0 = Factually perfect)")
    print(f"Reasoning: {metric.reason}\n")

    return metric.score

input_folder = "./output_feynman"
output_folder = "./result_evaluations_feynman"

os.makedirs(output_folder, exist_ok=True)

md_files = glob.glob(os.path.join(input_folder, "*.md"))

# this is a list of dict{topic:..., parsed_papers: list[ParsedText]}
parsed_papers_by_topic = parsing_all_the_papers()

checkpoint_path = Path("checkpoint_feynman.json")

if checkpoint_path.is_file():
    with open("checkpoint_feynman.json", "r") as f:
        results_evaluated = json.load(f)
else:
    results_evaluated = []

for file_path in md_files:
    file_name = os.path.basename(file_path)

    if file_name in results_evaluated:
        print(f"Already evaluated: {file_name}")
        continue

    print(f"Processing: {file_name}") 

    try:
        with open(file_path, "r") as f:
            print("OPENED FILE")
            file_content = f.read()
            topic, _ = os.path.splitext(file_name)

            print(f"CURRENT TOPIC: {topic}")
            print(file_content[:50])
            evaluation_result = EvaluationSummaries(
                topic = topic,
                model = "gpt_oss:120b",
                evaluations = []
            )

            topic_dict = next((d for d in parsed_papers_by_topic if d.get("topic") == topic), None)
            raw_papers_list = topic_dict["parsed_papers"]
            all_raw_papers = "\n".join([paper.parsed_text for paper in raw_papers_list])

            score = run_geval_test(research_gap_metric,all_raw_papers, file_content)
            evaluation_result.evaluations.append(score)

            filename = f"eval_{topic}_gpt_oss:120b.json"
            output_path = os.path.join(output_folder, filename)
            with open(output_path, "w") as f:
                f.write(evaluation_result.model_dump_json())

            results_evaluated.append(file_name)
            with open("checkpoint_SOTA_f_sum.json", "w") as f:
                f.write(json.dumps(results_evaluated))
        
    except Exception as e:
            print(f"Error processing {file_name}: {e}")