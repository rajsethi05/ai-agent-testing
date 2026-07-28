from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

Correctness_metric = GEval(name="Correctness",
                           criteria="Determine whether the actual output is factually correct based on the expected output.",
                           evaluation_steps=[
                               "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
                               "You should also heavily penalize omission of detail",
                               "Vague language, or contradicting OPINIONS, are OK", ],
                           evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT,
                                              LLMTestCaseParams.EXPECTED_OUTPUT], )

Consistency_metric = GEval(name="Consistency", evaluation_steps=[
    "Compare 'actual output' (first answer) and 'expected output' (second answer) for the same question.",
    "Penalise if the two answers state contradictory facts — e.g., one says X and the other says not-X.",
    "Penalise if key facts present in one answer are directly contradicted by the other.",
    "Minor differences in wording, phrasing, or level of detail are acceptable and should not be penalised.",
    "Award a high score if both answers convey the same core information without contradiction.", ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT], )
