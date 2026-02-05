from deepeval import evaluate
from deepeval.metrics import GEval, HallucinationMetric, AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

# from pathlib import Path
# from tools.pdf_parser_no_tool_version import parser

def test_test():

    test = LLMTestCase(
        input="quanto fa 4 diviso 2?",
        actual_output="3",
        expected_output="2"
    )

    metric_test = GEval(
        name="test",
        criteria = "controlla se la risposta alla domanda è corretta",
        evaluation_params=[LLMTestCaseParams.EXPECTED_OUTPUT,LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.7
    )
# folder_path = Path("./knowledge")
# inputs_papers = []
# for pdf_file in folder_path.glob("*.pdf"):
#     pdf_name = pdf_file.name
#     pdf_path = str(pdf_file)
#     parsed_text = parser(pdf_path)
#     inputs_papers.append(parsed_text)

# ## Summaries without judge 

# # 2. Define a G-Eval Metric for Technical Precision
# technical_precision = GEval(
#     name="Technical Precision",
#     criteria="""Check if the summary preserves the exact scientific findings. 
#                Are p-values, sample sizes, and chemical formulas accurate?""",
#     evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
#     threshold=0.7
# )

# # 3. Define a Hallucination Metric
# # This ensures the agent didn't "invent" findings not in the paper
# hallucination_check = HallucinationMetric(threshold=0.5, verbose_mode=True)
# relevancy = AnswerRelevancyMetric(threshold=0.6, verbose_mode=True)
# faithfulness = FaithfulnessMetric(threshold=0.6, verbose_mode=True)
# # 4. Prepare your Test Case
# # 'input' is the full paper text, 'actual_output' is your CrewAI summary
# test_case = LLMTestCase(
#     input="The full text of the scientific paper goes here...",
#     actual_output="The generated summary from your CrewAI flow goes here...",
#     context=["The full text of the scientific paper goes here..."] # Required for Hallucination
# )

# # 5. Run the evaluation
# # 'evaluate' will run all metrics on all test cases and print a summary table
# evaluate(
#     test_cases=[test_case],
#     metrics=[technical_precision, hallucination_check, relevancy, faithfulness],
#     print_results=True
# )