from deepeval import evaluate
from deepeval.metrics import GEval, HallucinationMetric, AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


# 2. Define a G-Eval Metric for Technical Precision
technical_precision = GEval(
    name="Technical Precision",
    criteria="""Check if the summary preserves the exact scientific findings. 
               Are p-values, sample sizes, and chemical formulas accurate?""",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7
)

# 3. Define a Hallucination Metric
# This ensures the agent didn't "invent" findings not in the paper
hallucination_check = HallucinationMetric(threshold=0.5)

# 4. Prepare your Test Case
# 'input' is the full paper text, 'actual_output' is your CrewAI summary
test_case = LLMTestCase(
    input="The full text of the scientific paper goes here...",
    actual_output="The generated summary from your CrewAI flow goes here...",
    context=["The full text of the scientific paper goes here..."] # Required for Hallucination
)

# 5. Run the evaluation
# 'evaluate' will run all metrics on all test cases and print a summary table
evaluate(
    test_cases=[test_case],
    metrics=[technical_precision, hallucination_check],
    print_results=True
)