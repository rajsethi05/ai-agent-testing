from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

from config import CONSISTENCY_THRESHOLD

Consistency_metric = GEval(name="Consistency", evaluation_steps=[
    "Compare 'actual output' (first answer) and 'expected output' (second answer) for the same question.",
    "Penalise if the two answers state contradictory facts — e.g., one says X and the other says not-X.",
    "Penalise if key facts present in one answer are directly contradicted by the other.",
    "Minor differences in wording, phrasing, or level of detail are acceptable and should not be penalised.",
    "Award a high score if both answers convey the same core information without contradiction.", ],
                           evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT,
                                              LLMTestCaseParams.EXPECTED_OUTPUT], threshold=CONSISTENCY_THRESHOLD)
