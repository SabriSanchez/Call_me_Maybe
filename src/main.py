import json
import argparse
import sys
from .tokenizer_utils import TokenizerUtils
from .decoder import RestrictedDecoder
from .engine import Engine
from .writer import Writer
from .parser import parse_prompts, parse_functions
from llm_sdk import Small_LLM_Model  # type: ignore


def main() -> None:
    """
    Main entry point for the Call-me-Maybe application.

    This function configures the command-line interface, parses input
    arguments, initializes the LLM model and components, orchestrates
    the function calling process via the Engine, and writes the
    resulting JSON output.

    It handles common errors such as missing input files or validation failures
    by printing descriptive error messages to stderr and exiting with a
    non-zero status code.
    """
    parser = argparse.ArgumentParser(description="Call-me-Maybe")

    parser.add_argument('--input',
                        nargs='?',
                        default='data/input/function_calling_tests.json'
                        )
    parser.add_argument('--functions_definition',
                        nargs='?',
                        default='data/input/functions_definition.json'
                        )
    parser.add_argument('--output',
                        nargs='?',
                        default='data/output/function_calling_results.json'
                        )

    args = parser.parse_args()

    try:
        with open(args.input, "r", encoding="utf-8") as f:
            raw_prompts = json.load(f)
        with open(args.functions_definition, "r", encoding="utf-8") as f:
            raw_functions = json.load(f)

        validated_prompts = parse_prompts(raw_prompts)
        validated_functions = parse_functions(raw_functions)

        if not validated_prompts or not validated_functions:
            print("Validation error: Incorrect input format.",
                  file=sys.stderr)
            sys.exit(1)

        model = Small_LLM_Model()
        tokenizer = TokenizerUtils(model)

        prompts_list = validated_prompts.model_dump()
        functions_list = validated_functions.model_dump()

        decoder = RestrictedDecoder(
            model,
            tokenizer,
            tokenizer.encode_functions(functions_list)
        )

        engine = Engine(model, decoder)
        results = engine.process(prompts_list, functions_list)
        writer = Writer(args.output)
        writer.write(results)

        print(f"Results successfully saved in {args.output}")

    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in input files: {e}",
              file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Error: File not found: {e.filename}",
              file=sys.stderr)
        sys.exit(1)
    except PermissionError as e:
        print(f"Error: Permission denied accessing file: {e.filename}",
              file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)
