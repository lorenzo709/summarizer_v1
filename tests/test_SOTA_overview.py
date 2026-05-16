from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval import assert_test

# 1. Define a metric that judges impact based ONLY on the topic
topic_impact_metric = GEval(
    name="Topic-Based Research Innovation",
    criteria="""
    Determine if the 'actual_output' provides innovative research opportunities 
    specifically for the given 'input' (the topic). 
    - The suggestions should be non-obvious.
    - They should address current trends in the field of the topic.
    - Penalize generic answers like 'more data is needed'.
    """,
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7
)

def test_topic_agent():
    # Your tool only takes this:
    topic = "Perovskite Solar Cells"
    
    # Your tool returns this:
    agent_output = "Researching long-term structural stability using encapsulated ionic liquids."

    # We map 'topic' to 'input' so the metric knows what the agent was trying to solve
    test_case = LLMTestCase(
        input=topic, 
        actual_output=agent_output
    )

    assert_test(test_case, [topic_impact_metric])