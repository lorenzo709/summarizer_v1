from deepeval import evaluate
from deepeval.metrics import GEval, HallucinationMetric, AnswerRelevancyMetric, FaithfulnessMetric, SummarizationMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from pathlib import Path
import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.pdf_parser_no_tool_version import parser
import json

# folder_path = Path("../knowledge")
# inputs_papers = []
# for pdf_file in folder_path.glob("*.pdf"):
#     pdf_name = pdf_file.name
#     pdf_path = str(pdf_file)
#     parsed_text = parser(pdf_path)
#     inputs_papers.append(parsed_text)

# current_dir = Path(__file__).parent
# json_path = current_dir / "data.json"
json_path = Path("./data_liquid_neural_networks.json")
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

gpt_summaries = data.get("summaries_gpt",[])
input_gpt = str("\n\n".join([item['summary'] for item in gpt_summaries if 'summary' in item] ))

output_gpt = str(data.get("final_summary_gpt",[]))

case_for_chat_gpt = LLMTestCase(
    input = input_gpt,
    actual_output= output_gpt
)

pipe_summaries = data.get("summaries_my_pipe",[])
input_pipe = str("\n\n".join([item['summary'] for item in pipe_summaries if 'summary' in item] ))

output_pipe = str(data.get("final_summary_my_pipe",[]))

case_for_pipe = LLMTestCase(
    input = input_pipe,
    actual_output= output_pipe
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

# summarization_metric = SummarizationMetric(
#     verbose_mode=True
# )

result_gpt = evaluate(
    test_cases=[case_for_chat_gpt],
    metrics=[synthesis_quality_metric],#summarization_metric],
    print_results=True,
    run_async=False,
    verbose_mode=True
)

result_my_pipe = evaluate(
    test_cases=[case_for_pipe],
    metrics=[synthesis_quality_metric],#summarization_metric],
    print_results=True,
    run_async=False,
    verbose_mode=True
)

print(f"TOPIC: {data['topic']}")