import json
from pathlib import Path
from typing import Any, TypedDict


class Writer:
    """
    Responsible for validating and recording the results of function calls
    to a file.

    Attributes:
        REQUIRED_KEYS (set): The set of keys required in each
        result dictionary.
        output_path (Path): The file path where results will be stored.
    """

    REQUIRED_KEYS = {"prompt", "name", "parameters"}

    class Result(TypedDict):
        """Defines the expected structure of a valid result dictionary."""
        prompt: str
        name: str
        parameters: dict[str, Any]

    def __init__(self, output_path: str) -> None:
        """
        Initializes the Writer with the destination output path.
        Args:
            output_path: The string path to the file where results will
            be saved.
        """
        self.output_path = Path(output_path)

    def _validate_result(self, result: dict[str, Any]) -> Result:
        """
        Validates the structure and types of an individual result dictionary.
        Args:
            result: The raw result dictionary to validate.
        Returns:
            Result: The validated result dictionary.
        Raises:
            TypeError: If the result is not a dictionary or contains
                    invalid types.
            ValueError: If the result is missing required keys or has
                    unexpected keys.
        """

        if not isinstance(result, dict):
            raise TypeError(
                f"Each result must be a dict, got {type(result)}"
            )

        if set(result.keys()) != self.REQUIRED_KEYS:
            raise ValueError(
                f"Invalid keys. Expected "
                f"{self.REQUIRED_KEYS}, "
                f"got {set(result.keys())}"
            )

        if not isinstance(result.get("prompt"), str):
            raise TypeError("prompt must be a string")

        if not isinstance(result.get("name"), str):
            raise TypeError("name must be a string")

        if not isinstance(result.get("parameters"), dict):
            raise TypeError("parameters must be a dictionary")

        return {
            "prompt": result["prompt"],
            "name": result["name"],
            "parameters": result["parameters"],
        }

    def _validate_results(
        self,
        results: list[dict[str, Any]]
    ) -> list[Result]:
        """
        Validates a list of result dictionaries.
        Args:
            results: A list of raw result dictionaries.
        Returns:
            list[Result]: A list of validated result dictionaries.
        Raises:
            TypeError: If the input is not a list.
        """

        if not isinstance(results, list):
            raise TypeError("results must be a list")

        return [
            self._validate_result(result)
            for result in results
        ]

    def write(self, results: list[dict[str, Any]]) -> None:
        """
        Validates the provided results and writes them to the output file in
        JSON format.
        This method automatically creates the parent directories if they do
        not exist.

        Args:
            results: The list of raw result dictionaries to be written.
        """

        validated_results = self._validate_results(results)

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            self.output_path,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                validated_results,
                f,
                indent=4,
                ensure_ascii=False
            )
