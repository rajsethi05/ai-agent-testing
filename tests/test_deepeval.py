from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from dotenv import load_dotenv

load_dotenv()
'''
Metrics Creation:
- name: Any name which we would like to call our metrics
- criteria: This is basically the instruction given to the LLM to tell it how to judge the output.
-  evaluation_steps: a list instructions outlining the exact steps the LLM should take for evaluation. If evaluation_steps is not provided, GEval will generate a series of evaluation_steps on your behalf based on the provided criteria.
- evaluation_params: a list of type LLMTestCaseParams. Include only the parameters that are relevant for evaluation
- threshold: passing threshold. As all the evaluation is scored in the range of 0-1

https://deepeval.com/docs/metrics-llm-evals#usage
'''
correctness_metric = GEval(
    name="Correctness",
    criteria="Determine whether the actual output is factually correct based on the expected output.",
    # NOTE: you can only provide either criteria or evaluation_steps, and not both
    evaluation_steps=[
        "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
        "You should also heavily penalize omission of detail",
        "Vague language, or contradicting OPINIONS, are OK"
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)


# def test_correctness():
#     test_case = LLMTestCase(
#         input="I have a persistent cough and fever. Should I be worried?",
#         # Replace this with the actual output from your LLM application
#         actual_output="See a doctor if symptoms worsen or don't improve in a few days.",
#         # actual_output="A persistent cough and fever could be a viral infection or something more serious. See a doctor if symptoms worsen or don't improve in a few days.",
#         expected_output="A persistent cough and fever could indicate a range of illnesses, from a mild viral infection to more serious conditions like pneumonia or COVID-19. You should seek medical attention if your symptoms worsen, persist for more than a few days, or are accompanied by difficulty breathing, chest pain, or other concerning signs."
#     )
#     assert_test(test_case, [correctness_metric])