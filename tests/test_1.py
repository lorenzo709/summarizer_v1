from deepeval import evaluate
from deepeval.metrics import GEval, HallucinationMetric, AnswerRelevancyMetric, FaithfulnessMetric, SummarizationMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from pathlib import Path
import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.pdf_parser_no_tool_version import parser
import json

# pro_con_metric = GEval(
#     name="Pros and Cons",
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
folder_path = Path("../knowledge")
inputs_papers = []
for pdf_file in folder_path.glob("*.pdf"):
    pdf_name = pdf_file.name
    pdf_path = str(pdf_file)
    parsed_text = parser(pdf_path)
    inputs_papers.append(parsed_text)

# current_dir = Path(__file__).parent
# json_path = current_dir / "data.json"
json_path = Path("./data.json")
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# summarization_metric = SummarizationMetric(
#     threshold=0.5,
# )
summarization_metric = GEval(
        name="Summary Quality",
        # Criteria defines what the LLM should look for
        criteria="Determine if the summary captures the core methodology and limitations of the paper without hallucinating facts.",
        # Evaluation steps guide the LLM's reasoning process (CoT)
        evaluation_steps=[
            "Esure that core methodology is included in the summary.",
            "Verify if at least one limitation or 'gap' is identified in the summary.",
            "Ensure no information is included that wasn't in the original text.",
            "Assess the professional tone and academic clarity."
        ],
        # Which parts of the test case should the judge look at?
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.5
    )
## Summaries without judge 
test_cases_no_judges = []

print(len(inputs_papers))
print(len(data))
for i in range(len(inputs_papers)):
    case = LLMTestCase(
        input = str(inputs_papers[i]),
        actual_output= str(data["single_papers_no_judge"][i]["summary"])
    )
    test_cases_no_judges.append(case)

## Summaries with judge 
test_cases_with_judge = []

for i in range(len(inputs_papers)):
    case = LLMTestCase(
        input = str(inputs_papers[i]),
        actual_output= data["single_paper_judge"][i]["summary"]
    )
    test_cases_with_judge.append(case)

result_no_judge = evaluate(
    test_cases=test_cases_no_judges,
    metrics=[summarization_metric],
    print_results=True,
    run_async=False,
    verbose_mode=True
)

print(len(test_cases_with_judge))
result_with_judge = evaluate(
    test_cases=test_cases_with_judge,
    metrics=[summarization_metric],
    print_results=True,
    run_async=False,
    verbose_mode=True
)

## TESTING PROS AND CONS

pros_cons_metric = GEval(
    name="Pros and Cons Depth & Accuracy",
    criteria="""Determine if the output provides a series of distinct points for both 
                'Main Benefits' and 'Areas for Improvement'. The points must be 
                technically grounded in the paper and avoid vague language.""",
    evaluation_steps=[
        "Check if the output is divided into 'Main Benefits' and 'Areas for Improvement' (or similar headings).",
        "Verify that each point is a discrete, specific insight rather than a broad generalization.",
        "Ensure 'Main Benefits' focus on the novel contributions or strengths of the methodology.",
        "Ensure 'Areas for Improvement' identify legitimate technical gaps, limitations, or potential refinements.",
        "Count the points: A high-quality response should provide at least 3 distinct points per section if the paper depth allows."
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
)

test_cases_single_pro_cons = []
for i in range(len(inputs_papers)):
    case = LLMTestCase(
        input = str(inputs_papers[i]),
        actual_output= str(data["single_papers_no_judge"][i]["pros_cons"])
    )
    test_cases_single_pro_cons.append(case)

result_pro_cons_no_judge = evaluate(
    test_cases=test_cases_single_pro_cons,
    metrics=[pros_cons_metric],
    print_results=True,
    run_async=False,
    verbose_mode=True
)