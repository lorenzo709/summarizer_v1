from deepeval import evaluate
from deepeval.metrics import GEval, HallucinationMetric, AnswerRelevancyMetric, FaithfulnessMetric, SummarizationMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from pathlib import Path
import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.pdf_parser_no_tool_version import parser
import json

# correctness_metric = GEval(
#     name="Correctness",
#     criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
#     evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
#     threshold=0.5
# )

# test_case = LLMTestCase(
#     input="I have a persistent cough and fever. Should I be worried?",
#     # Replace this with the actual output from your LLM application
#     actual_output="A persistent cough and fever could signal various illnesses, from minor infections to more serious conditions like pneumonia or COVID-19. It's advisable to seek medical attention if symptoms worsen, persist beyond a few days, or if you experience difficulty breathing, chest pain, or other concerning signs.",
#     expected_output="A persistent cough and fever could indicate a range of illnesses, from a mild viral infection to more serious conditions like pneumonia or COVID-19. You should seek medical attention if your symptoms worsen, persist for more than a few days, or are accompanied by difficulty breathing, chest pain, or other concerning signs."
# )

# evaluate([test_case], [correctness_metric])
folder_path = Path("./knowledge")
inputs_papers = []
for pdf_file in folder_path.glob("*.pdf"):
    pdf_name = pdf_file.name
    pdf_path = str(pdf_file)
    parsed_text = parser(pdf_path)
    inputs_papers.append(parsed_text)

current_dir = Path(__file__).parent
json_path = current_dir / "data.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

summarization_metric = SummarizationMetric(
    threshold=0.5,
)
## Summaries without judge 
test_cases_no_judges = []

for i in range(len(inputs_papers)):
    case = LLMTestCase(
        input = inputs_papers[i],
        actual_output= data["single_papers_no_judge"][i]["summary"]
    )
    test_cases_no_judges.append(case)

test_cases_with_judge = []

# for i in range(len(inputs_papers)):
#     case = LLMTestCase(
#         input = inputs_papers[i],
#         actual_output= data["single_paper_judge"][i]["summary"]
#     )
#     test_cases_with_judge.append(case)


print(len(test_cases_no_judges))
result_no_judge = evaluate(
    test_cases=[test_cases_no_judges],
    metrics=[summarization_metric],
    print_results=True
)

# print(len(test_cases_with_judge))
# result_with_judge = evaluate(
#     test_cases=[test_cases_with_judge],
#     metrics=[summarization_metric],
#     print_results=True
# )

