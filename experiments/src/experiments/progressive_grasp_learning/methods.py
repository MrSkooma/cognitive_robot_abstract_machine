"""
The template circuit and the methods that learn a model of a new grasp task from a few
attempts, with or without a task learned before it.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import jax
import numpy as np
from numpy.typing import NDArray
from typing_extensions import List

from experiments.progressive_grasp_learning.circuit_variables import (
    GraspOutcomeVariables,
    LiftOutcome,
)
from experiments.progressive_grasp_learning.scoring import OutcomeLikelihoods
from experiments.progressive_grasp_learning.task_attempts import ScaledAttempts
from krrood.ormatic.utils import classproperty
from probabilistic_model.adapters.circuit_representations import CircuitRepresentations
from probabilistic_model.distributions.gaussian import GaussianDistribution
from probabilistic_model.learning.expectation_maximization import (
    ExpectationMaximization,
)
from probabilistic_model.learning.progressive.expectation_maximization import (
    ProgressiveExpectationMaximization,
)
from probabilistic_model.learning.progressive.progressive_circuit import (
    CircuitColumn,
    EarlierColumnTarget,
    ProgressiveProbabilisticCircuit,
)
from probabilistic_model.learning.region_graph.region_graph import RegionGraph
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    LeafUnit,
    ProbabilisticCircuit,
)

# %% the template


@dataclass(kw_only=True)
class RatSpnTemplate:
    """
    A RAT-SPN over the grasp and its outcome: a random region graph turned into a
    circuit, with start parameters drawn independently of any data.
    """

    depth: int
    """
    How often each repetition splits the variables.
    """

    units_per_region: int
    """
    The number of input units and of sum units of each region.
    """

    repetitions: int = 2
    """
    How many random splittings of the variables the region graph holds.
    """

    partitions_per_region: int = 2
    """
    Into how many regions each split divides a region.
    """

    start_scale: float = 0.25
    """
    The standard deviation every Gaussian leaf starts with.
    """

    def circuit(
        self, *, variables: GraspOutcomeVariables, seed: int
    ) -> ProbabilisticCircuit:
        """
        :param variables: The variables of the circuit.
        :param seed: Seeds the region graph and the start parameters, so the same seed
            gives the same circuit.
        :return: The circuit, with the mean of every Gaussian leaf drawn uniformly from 0
            to 1 and its standard deviation :attr:`start_scale`.

        .. note::

            The region graph draws its splits from numpy's global random generator,
            which this method seeds.
        """
        np.random.seed(seed)
        region_graph = RegionGraph(
            variables.variables,
            partitions=self.partitions_per_region,
            depth=self.depth,
            repetitions=self.repetitions,
        ).create_random_region_graph()
        circuit = CircuitRepresentations().convert(
            region_graph.as_probabilistic_circuit(
                input_units=self.units_per_region,
                sum_units=self.units_per_region,
                key=jax.random.PRNGKey(seed),
            ),
            ProbabilisticCircuit,
        )
        generator = np.random.default_rng(seed)
        for unit in circuit.nodes():
            if isinstance(unit, LeafUnit) and isinstance(
                unit.distribution, GaussianDistribution
            ):
                unit.distribution.location = float(generator.uniform())
                unit.distribution.scale = self.start_scale
        return circuit


# %% settings of learning


@dataclass(kw_only=True)
class LearningSettings:
    """
    How a column of a progressive circuit is built and learned.
    """

    template: RatSpnTemplate
    """
    The circuit every column is a copy of.
    """

    earlier_column_target: EarlierColumnTarget = EarlierColumnTarget.ALIGNED_SUM_UNIT
    """
    The units of earlier columns a new column's sum units get edges to.
    """

    earlier_column_share: float = 0.5
    """
    The share of each new sum unit's start weight on the edges to earlier columns.
    """

    iterations: int = 20
    """
    How many iterations of expectation maximization learn a column.
    """

    minimum_gaussian_variance: float = 0.02**2
    """
    The lowest variance a learned Gaussian leaf may have.
    """

    def progressive_circuit(
        self, *, variables: GraspOutcomeVariables, seed: int
    ) -> ProgressiveProbabilisticCircuit:
        """
        :param variables: The variables of the circuit.
        :param seed: Seeds the template.
        :return: A progressive circuit without columns.
        """
        return ProgressiveProbabilisticCircuit(
            template=self.template.circuit(variables=variables, seed=seed),
            earlier_column_share=self.earlier_column_share,
            earlier_column_target=self.earlier_column_target,
        )

    def learn(
        self,
        *,
        progressive_circuit: ProgressiveProbabilisticCircuit,
        column: CircuitColumn,
        attempts: ScaledAttempts,
        variables: GraspOutcomeVariables,
    ) -> None:
        """
        Learn a column from attempts.

        :param progressive_circuit: The circuit holding the column.
        :param column: The column to learn.
        :param attempts: The attempts it learns from.
        :param variables: The variables of the circuit.
        """
        ProgressiveExpectationMaximization(
            progressive_circuit,
            expectation_maximization=ExpectationMaximization(
                minimum_gaussian_variance=self.minimum_gaussian_variance
            ),
        ).learn(
            variables.rows(
                grasps=attempts.grasps, outcomes=attempts.lifted.astype(np.int64)
            ),
            column,
            iterations=self.iterations,
        )


# %% models


class OutcomeModel(ABC):
    """
    A model that gives grasps a joint likelihood with each outcome.
    """

    @abstractmethod
    def outcome_likelihoods(self, grasps: NDArray[np.float64]) -> OutcomeLikelihoods:
        """
        :param grasps: One scaled grasp per row.
        :return: The joint log-likelihood of each grasp with each outcome.
        """


@dataclass(kw_only=True)
class ColumnModel(OutcomeModel):
    """
    One column of a progressive circuit, used alone as the model of its task.
    """

    progressive_circuit: ProgressiveProbabilisticCircuit
    """
    The circuit holding the column.
    """

    column: CircuitColumn
    """
    The column.
    """

    variables: GraspOutcomeVariables
    """
    The variables of the circuit.
    """

    def outcome_likelihoods(self, grasps: NDArray[np.float64]) -> OutcomeLikelihoods:
        self.progressive_circuit.restrict_root_to(self.column)
        return OutcomeLikelihoods(
            log_joint=np.column_stack(
                [
                    self.progressive_circuit.circuit.log_likelihood(
                        self.variables.rows(
                            grasps=grasps,
                            outcomes=np.full(len(grasps), outcome, dtype=np.int64),
                        )
                    )
                    for outcome in LiftOutcome
                ]
            )
        )


# %% a task learned before


@dataclass(kw_only=True)
class EarlierTask:
    """
    A task learned before the new one, as the only column of a progressive circuit.
    """

    progressive_circuit: ProgressiveProbabilisticCircuit
    """
    The circuit holding the column.
    """

    column: CircuitColumn
    """
    The column of the task.
    """

    @classmethod
    def learned(
        cls,
        *,
        attempts: ScaledAttempts,
        settings: LearningSettings,
        variables: GraspOutcomeVariables,
        seed: int,
    ) -> EarlierTask:
        """
        :param attempts: The attempts of the task.
        :param settings: How the column is built and learned.
        :param variables: The variables of the circuit.
        :param seed: Seeds the template.
        :return: The task, learned from its attempts.
        """
        progressive_circuit = settings.progressive_circuit(
            variables=variables, seed=seed
        )
        column = progressive_circuit.add_column(task_name="earlier task")
        settings.learn(
            progressive_circuit=progressive_circuit,
            column=column,
            attempts=attempts,
            variables=variables,
        )
        return cls(progressive_circuit=progressive_circuit, column=column)


# %% methods


class TransferMethod(ABC):
    """
    A way to learn a model of a new task from its attempts, which may use a task learned
    before it.
    """

    @property
    @abstractmethod
    def reads_earlier_task(self) -> bool:
        """
        Whether the model depends on the earlier task.
        """

    @abstractmethod
    def model(
        self, *, training: ScaledAttempts, earlier: EarlierTask, seed: int
    ) -> OutcomeModel:
        """
        :param training: The attempts of the new task to learn from.
        :param earlier: The task learned before, from the same seed.
        :param seed: Seeds the template of a column built here.
        :return: The model of the new task.
        """


@dataclass(kw_only=True)
class ProgressiveColumn(TransferMethod):
    """
    A new column added to the earlier task's progressive circuit, learned while the
    earlier column stays unchanged.
    """

    settings: LearningSettings
    """
    How the new column is learned; the earlier task was built with the same settings.
    """

    variables: GraspOutcomeVariables = field(default_factory=GraspOutcomeVariables)
    """
    The variables of the circuit.
    """

    @property
    def reads_earlier_task(self) -> bool:
        return True

    def model(
        self, *, training: ScaledAttempts, earlier: EarlierTask, seed: int
    ) -> OutcomeModel:
        progressive_circuit = copy.deepcopy(earlier.progressive_circuit)
        column = progressive_circuit.add_column(task_name="new task")
        self.settings.learn(
            progressive_circuit=progressive_circuit,
            column=column,
            attempts=training,
            variables=self.variables,
        )
        return ColumnModel(
            progressive_circuit=progressive_circuit,
            column=column,
            variables=self.variables,
        )


@dataclass(kw_only=True)
class FixedScratch(TransferMethod):
    """
    A copy of the template learned on the new task's attempts alone.
    """

    settings: LearningSettings
    """
    How the column is built and learned.
    """

    variables: GraspOutcomeVariables = field(default_factory=GraspOutcomeVariables)
    """
    The variables of the circuit.
    """

    @property
    def reads_earlier_task(self) -> bool:
        return False

    def model(
        self, *, training: ScaledAttempts, earlier: EarlierTask, seed: int
    ) -> OutcomeModel:
        return self.learned_column(training=training, seed=seed)

    def learned_column(self, *, training: ScaledAttempts, seed: int) -> ColumnModel:
        """
        :param training: The attempts to learn from.
        :param seed: Seeds the template.
        :return: The column learned on the attempts alone.
        """
        progressive_circuit = self.settings.progressive_circuit(
            variables=self.variables, seed=seed
        )
        column = progressive_circuit.add_column(task_name="new task")
        self.settings.learn(
            progressive_circuit=progressive_circuit,
            column=column,
            attempts=training,
            variables=self.variables,
        )
        return ColumnModel(
            progressive_circuit=progressive_circuit,
            column=column,
            variables=self.variables,
        )


@dataclass(kw_only=True)
class TunedScratch(TransferMethod):
    """
    Fixed scratch with the settings, among candidates, that predict lifting best in
    cross-validation on the new task's attempts.
    """

    candidates: List[LearningSettings]
    """
    The settings to choose from; on a tie the earlier one is chosen.
    """

    variables: GraspOutcomeVariables = field(default_factory=GraspOutcomeVariables)
    """
    The variables of the circuits.
    """

    number_of_folds: int = 5
    """
    The number of folds when the attempts are too many for leaving one out.
    """

    @classproperty
    def most_attempts_left_out_one_by_one(cls) -> int:
        """
        The most attempts cross-validation leaves out one at a time.
        """
        return 10

    @property
    def reads_earlier_task(self) -> bool:
        return False

    def model(
        self, *, training: ScaledAttempts, earlier: EarlierTask, seed: int
    ) -> OutcomeModel:
        losses = [
            self.cross_validated_loss(settings=settings, training=training, seed=seed)
            for settings in self.candidates
        ]
        return FixedScratch(
            settings=self.candidates[int(np.argmin(losses))], variables=self.variables
        ).learned_column(training=training, seed=seed)

    def cross_validated_loss(
        self, *, settings: LearningSettings, training: ScaledAttempts, seed: int
    ) -> float:
        """
        :param settings: The candidate settings.
        :param training: The attempts to cross-validate on.
        :param seed: Seeds the template.
        :return: The mean clipped log loss of the held-out attempts' outcomes.
        """
        number_of_attempts = len(training.lifted)
        folds = np.arange(number_of_attempts)
        if number_of_attempts > self.most_attempts_left_out_one_by_one:
            folds = folds % self.number_of_folds
        scratch = FixedScratch(settings=settings, variables=self.variables)
        losses = []
        for fold in np.unique(folds):
            held_out = folds == fold
            model = scratch.learned_column(training=training.rows(~held_out), seed=seed)
            losses.append(
                model.outcome_likelihoods(training.grasps[held_out]).clipped_log_losses(
                    training.lifted[held_out]
                )
            )
        return float(np.mean(np.concatenate(losses)))


@dataclass(kw_only=True)
class EarlierColumnUnchanged(TransferMethod):
    """
    The earlier task's column used as the model of the new task, without learning.
    """

    variables: GraspOutcomeVariables = field(default_factory=GraspOutcomeVariables)
    """
    The variables of the circuit.
    """

    @property
    def reads_earlier_task(self) -> bool:
        return True

    def model(
        self, *, training: ScaledAttempts, earlier: EarlierTask, seed: int
    ) -> OutcomeModel:
        return ColumnModel(
            progressive_circuit=earlier.progressive_circuit,
            column=earlier.column,
            variables=self.variables,
        )
