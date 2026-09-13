import re
from dataclasses import dataclass


ABSTENTION_MESSAGE = (
    "The indexed documents do not provide enough information."
)

CITATION_PATTERN = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class CitationValidationResult:
    is_valid: bool
    cited_source_numbers: list[int]
    warnings: list[str]


def validate_answer_citations(
    answer: str,
    available_source_count: int,
) -> CitationValidationResult:
    """
    Validate that an answer cites only retrieved source numbers.

    This checks citation structure and availability. It does not
    determine whether each cited passage logically supports the claim.
    """

    normalized_answer = answer.strip()

    if normalized_answer == ABSTENTION_MESSAGE:
        return CitationValidationResult(
            is_valid=True,
            cited_source_numbers=[],
            warnings=[],
        )

    cited_numbers = sorted(
        {
            int(match)
            for match in CITATION_PATTERN.findall(answer)
        }
    )

    warnings: list[str] = []

    if not cited_numbers:
        warnings.append(
            "The generated answer contains no source citations."
        )

    invalid_numbers = [
        number
        for number in cited_numbers
        if number < 1 or number > available_source_count
    ]

    if invalid_numbers:
        invalid_text = ", ".join(
            str(number) for number in invalid_numbers
        )

        warnings.append(
            f"The answer cites unavailable sources: {invalid_text}."
        )

    return CitationValidationResult(
        is_valid=not warnings,
        cited_source_numbers=cited_numbers,
        warnings=warnings,
    )