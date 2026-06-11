from deepeval import evaluate
from deepeval.metrics import GEval, HallucinationMetric, AnswerRelevancyMetric, FaithfulnessMetric, SummarizationMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

import os 
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "..", "src")
tool_path = os.path.join(current_dir, "..", "tools")
sys.path.append(os.path.abspath(src_path))
sys.path.append(os.path.abspath(tool_path))

from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MyTypes import PaperFound, ParsedText
from pdf_parser_no_tool_version import parser
import json

def parsing_all_the_papers():

    knowledge_folders =["../knowledge_chatgpt/catalytic water splitting on platinum",
                         "../knowledge_chatgpt/liquid neural networks continuous-time signal processing",
                         "../knowledge_chatgpt/Retrival-Augmented Generation for Legacy Code Refactoring",
                         "../knowledge_chatgpt/Visual Transformer",
                         "../knowledge_chatgpt/Zero-shot Robot Manipulation via CLIP-based Spatial Reasoning"]

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

json_path = Path("./data_RAG_for_legacy_code.json")
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

gpt_summaries = data.get("summaries_gpt",[])
paper_parsed_by_topic = parsing_all_the_papers()

topic_dict = next((d for d in paper_parsed_by_topic if d.get("topic") == data['topic']), None)
raw_papers_list = topic_dict["parsed_papers"]
all_raw_papers = "\n".join([paper.parsed_text for paper in raw_papers_list])

input_gpt_summaries = str("\n\n".join([item['summary'] for item in gpt_summaries if 'summary' in item and "PAPER" not in item['summary']] ))

final_input_gpt = "\n\n".join([all_raw_papers, input_gpt_summaries])
output_gpt = str(data.get("final_summary_gpt",[]))

case_for_chat_gpt = LLMTestCase(
    input = final_input_gpt,
    actual_output= output_gpt
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

result_gpt = evaluate(
    test_cases=[case_for_chat_gpt],
    metrics=[synthesis_quality_metric],#summarization_metric],
    print_results=True,
    run_async=False,
    verbose_mode=True
)
print(f"TOPIC: {data['topic']}")