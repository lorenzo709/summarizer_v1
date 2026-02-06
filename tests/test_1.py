from deepeval import evaluate
from deepeval.metrics import GEval, HallucinationMetric, AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

# from pathlib import Path
# from tools.pdf_parser_no_tool_version import parser

correctness_metric = GEval(
    name="Correctness",
    criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=0.5
)

test_case = LLMTestCase(
    input="I have a persistent cough and fever. Should I be worried?",
    # Replace this with the actual output from your LLM application
    actual_output="A persistent cough and fever could signal various illnesses, from minor infections to more serious conditions like pneumonia or COVID-19. It's advisable to seek medical attention if symptoms worsen, persist beyond a few days, or if you experience difficulty breathing, chest pain, or other concerning signs.",
    expected_output="A persistent cough and fever could indicate a range of illnesses, from a mild viral infection to more serious conditions like pneumonia or COVID-19. You should seek medical attention if your symptoms worsen, persist for more than a few days, or are accompanied by difficulty breathing, chest pain, or other concerning signs."
)

evaluate([test_case], [correctness_metric])
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