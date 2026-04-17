"""Tools for hallucination analysis with attribution graphs and activation steering."""

from .controller import NeuralActivationSteerer, NeuralControllerState, train_hallucination_controller
from .dataset import FactExample, answer_is_correct, build_messages, build_prompt, load_dataset
from .steering import ActivationSteerer, SteeringState, fit_steering_state

__all__ = [
    "ActivationSteerer",
    "FactExample",
    "NeuralActivationSteerer",
    "NeuralControllerState",
    "SteeringState",
    "answer_is_correct",
    "build_messages",
    "build_prompt",
    "fit_steering_state",
    "load_dataset",
    "train_hallucination_controller",
]
