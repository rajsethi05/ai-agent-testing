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

Groundedness_metric = GEval(name="Groundedness", evaluation_steps=[
    "Check if the answer introduces any information, conclusions, or recommendations that are not explicitly present in the retrieval context.",
    "Penalise answers that extrapolate or draw inferences beyond what the context directly states.",
    "Penalise answers that use generalisations such as 'generally', 'typically', 'experts say', or 'studies show' when the context does not support such claims.",
    "Reward answers that accurately reflect the scope and limitations of the provided context, including saying 'I don't know' when the context is insufficient.", ],
                            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT,
                                               LLMTestCaseParams.RETRIEVAL_CONTEXT], )