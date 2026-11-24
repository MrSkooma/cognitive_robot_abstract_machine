"""
Store the parts of objects of the pick-up experiment, in the directory ``--stored-
parts`` reads, so that no later run has to locate them again.

The parts are located either by the vision models on this machine's CPU or from
answers of the part finding server recorded before. Run it with
``python experiments/scripts/grasp_learning/store_parts.py``; see ``--help``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from experiments.physical_pick_up.objects import PickUpObject
from experiments.physical_pick_up.scene import PickUpScene
from experiments.physical_pick_up.stored_parts import (
    AnswerStoringLocator,
    RecordedAnswerLocator,
)
from semantic_digital_twin.adapters.part_finding.server import PartFindingCache
from semantic_digital_twin.adapters.part_finding.vision_language_models import (
    Device,
    QwenBoxDetector,
    QwenModel,
    Sam2BoxSegmenter,
)
from semantic_digital_twin.part_finding.annotation_part_finder import (
    AnnotationPartFinder,
)
from semantic_digital_twin.part_finding.locating import PartLocator
from semantic_digital_twin.part_finding.vision_language import (
    VisionLanguagePartLocator,
)

# %% storing the parts


@dataclass(kw_only=True)
class PartStorage:
    """
    Locates the parts of objects once and stores the answers.
    """

    answering_locator: PartLocator
    """
    The locator whose answers are stored.
    """

    stored_parts_directory: Path
    """
    Where the stored parts go.
    """

    def store_parts(self, *, pick_up_object: PickUpObject) -> int:
        """
        Build the scene of an object, which asks for its parts, and store the answers.

        :param pick_up_object: The object whose parts are stored.
        :return: How many answers were stored for it.
        """
        locator = AnswerStoringLocator(
            answering_locator=self.answering_locator,
            directory=self.stored_parts_directory,
            object_name=pick_up_object.name.lower(),
        )
        PickUpScene(
            object_description=pick_up_object.value,
            part_finder=AnnotationPartFinder(locator=locator),
        )
        return locator.stored_answer_count


# %% the command line


def main() -> None:
    """
    Store the parts of the objects named on the command line.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recorded-answers",
        type=Path,
        help="a directory of recorded answers, one file per query; without it the "
        "vision models locate the parts on this machine's CPU",
    )
    parser.add_argument(
        "--model",
        choices=[model.name.lower() for model in QwenModel],
        default=QwenModel.QWEN3_VL_4B.name.lower(),
        help="the vision-language model that finds the parts on the CPU",
    )
    parser.add_argument(
        "--stored-parts",
        type=Path,
        required=True,
        help="the directory the stored parts go to",
    )
    parser.add_argument(
        "--objects",
        nargs="+",
        required=True,
        choices=[pick_up_object.name.lower() for pick_up_object in PickUpObject],
        help="the objects whose parts are stored",
    )
    arguments = parser.parse_args()
    if arguments.recorded_answers is None:
        answering_locator = VisionLanguagePartLocator(
            detector=QwenBoxDetector(
                model=QwenModel[arguments.model.upper()], device=Device.CPU
            ),
            segmenter=Sam2BoxSegmenter(device=Device.CPU),
        )
    else:
        answering_locator = RecordedAnswerLocator(
            recorded_answers=PartFindingCache(directory=arguments.recorded_answers)
        )
    storage = PartStorage(
        answering_locator=answering_locator,
        stored_parts_directory=arguments.stored_parts,
    )
    for object_name in arguments.objects:
        stored = storage.store_parts(pick_up_object=PickUpObject[object_name.upper()])
        print(f"{object_name}: {stored} answers stored")


if __name__ == "__main__":
    main()
