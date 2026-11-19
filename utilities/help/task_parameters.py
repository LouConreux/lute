import argparse
import logging
import os
import sys
from typing import Dict, Optional, List, Set, Tuple, Any, Callable, Generic, Union, cast
from typing_extensions import TypedDict, TypeVar

import pprint

import lute.io.models
from lute.execution.executor import BaseExecutor
from lute.io.models.base import TaskParameters
from lute import managed_tasks

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


T = TypeVar("T")


class OptionalDictKey(Generic[T]): ...


class PropertyDict(TypedDict):
    default: str
    description: str
    title: str
    type: OptionalDictKey[str]
    anyOf: OptionalDictKey[List[Dict[str, str]]]  # Either an anyOf or type per property
    # Generally only for ThirdPartyTasks
    rename_param: Optional[str]
    flag_type: Optional[str]
    # Other additional field attributes
    env_names: Optional[Set[str]]
    maxLength: Optional[int]
    minLength: Optional[int]
    format: Optional[str]
    exclusiveMinimum: Optional[int]
    minimum: Optional[int]


class ObjectDefintion(TypedDict):
    description: str
    properties: Dict[str, PropertyDict]
    title: str
    type: str


class ModelSchema(TypedDict):
    definitions: Optional[Dict[str, ObjectDefintion]]
    description: str
    properties: Dict[str, PropertyDict]
    required: Optional[List[str]]
    title: str
    type: str


class EnumSpec(TypedDict):
    enum: List[Union[str, int, bool]]
    type: str


class EnvVarDefinition(TypedDict):
    default: str
    value: Optional[str]
    description: str


LUTE_ENV_VARS: Dict[str, EnvVarDefinition] = {
    "LUTE_USE_ZMQ": {
        "default": "1",
        "value": os.getenv("LUTE_USE_ZMQ"),
        "description": (
            "  - Toggles between the use of a socket-based and ZMQ-based IPC backend.\n"
            "  - These are functionally identical, but the raw-socket option allows non-ZMQ installs.\n"
        ),
    },
    "LUTE_USE_TCP": {
        "default": "1",
        "value": os.getenv("LUTE_USE_TCP"),
        "description": (
            "  - Toggles between the use of TCP and Unix IPC sockets.\n"
            "  - The Unix sockets can be faster if only using 1 node.\n"
            "  - If multiple nodes are in use, then TCP cannot be turned off.\n"
        ),
    },
    "LUTE_DB_SPEC_VERSION": {
        "default": "2",
        "value": os.getenv("LUTE_DB_SPEC_VERSION"),
        "description": (
            "  - Selects the database specification to use for the lute.db.\n"
            "  - Generally, the latest should always be used unless needing to interface with old data.\n"
            "  - Version 3 (labels/units for multi-run processing) is in development.\n"
        ),
    },
    "LUTE_MAESTRO_LOG_LEVEL": {
        "default": "info for all loggers",
        "value": os.getenv("LUTE_MAESTRO_LOG_LEVEL"),
        "description": (
            "  - Sets the log level of the `maestro` workflow manager.\n"
            "  - It can be set per logger, or for all loggers.\n"
            "    - E.g. `LUTE_MAESTRO_LOG_LEVEL=debug`\n"
            "    - Or LUTE_MAESTRO_LOG_LEVEL=LWM:Manager=trace.\n"
        ),
    },
    "LUTE_DEBUG_EXIT_AT_YAML": {
        "default": "Not set",
        "value": os.getenv("LUTE_DEBUG_EXIT_AT_YAML"),
        "description": (
            "  - If set at all, exit after parsing YAML.\n"
            "  - This variable enables an early debug exit.\n"
            "  - The YAML will be parsed and any template substitutions evaluated.\n"
            "  - It will then print and exit.\n"
        ),
    },
    "LUTE_DEBUG_BEFORE_TPP_EXEC": {
        "default": "Not set",
        "value": os.getenv("LUTE_DEBUG_BEFORE_TPP_EXEC"),
        "description": (
            "  - If set at all, exit before submitting third-party Task.\n"
            "  - This variable enables an early debug exit.\n"
            "  - The submission command will be printed for a third-party Task.\n"
            "  - It will exit after printing before actually submitting the Task.\n"
        ),
    },
    "LUTE_MAESTRO_HEARTBEAT": {
        "default": "30 seconds period",
        "value": os.getenv("LUTE_MAESTRO_HEARTBEAT"),
        "description": (
            "  - The period at which the Executor sends sign of life updates to the "
            "`maestro` workflow manager.\n"
            "  - The heartbeat period is used to judge whether a managed Task has "
            "become unresponsive.\n"
            "  - The value is set in seconds.\n"
        ),
    },
    "LUTE_NO_HEARTBEAT_TTL": {
        "default": "120 seconds period",
        "value": os.getenv("LUTE_NO_HEARTBEAT_TTL"),
        "description": (
            "  - Sets the time after which `maestro` will use a fallback strategy to "
            "check job status.\n"
            "  - Fallback strategy is used if it has not received a heartbeat update.\n"
            "  - E.g., when using a SLURM Launcher, `maestro` will check the SLURMDB.\n"
            "  - If the fallback strategy shows the job has exited, the job will be marked "
            "as failed.\n"
            "  - The value is set in seconds.\n"
        ),
    },
    "LUTE_NO_HEARTBEAT_KILL": {
        "default": "900 seconds period (15 minutes)",
        "value": os.getenv("LUTE_NO_HEARTBEAT_KILL"),
        "description": (
            "  - Sets the time after which an unresponsive job will be killed.\n"
            "  - This applies to both the Execution and maestro levels.\n"
            "    - The Executor monitors the Task for responsiveness.\n"
            "    - `maestro` monitors the Executor.\n"
            "  - The value is set in seconds.\n"
        ),
    },
}


def _format_enum(param: str, enum_spec: EnumSpec) -> str:
    indent: str = " " * (len(param) + 2 + 5)
    output: str = "Enum["
    for item in enum_spec["enum"]:
        if output != "Enum[":
            output += f",\n{indent}"
        output += str(item)
    output += "]"
    if "type" in enum_spec:
        output += f"({enum_spec['type']})"

    return output


def _format_parameter_row(
    param: str,
    param_description: PropertyDict,
    validators: Optional[List[Callable]] = None,
) -> str:
    """Take a property dictionary for a parameter and format it for printing."""
    # Either type or anyOf exists but not both
    typeinfo: Union[str, OptionalDictKey]
    if "type" in param_description:
        typeinfo = param_description["type"]
    elif "anyOf" in param_description:  # anyOf is present instead
        typeinfo = ""
        for item in param_description["anyOf"]:
            if "enum" in item:
                if typeinfo:
                    typeinfo += "\n" + " " * (len(param) - 1) + " | "
                typeinfo += _format_enum(param, item)
    elif "allOf" in param_description and "$ref" in param_description["allOf"][0]:
        typeinfo = param_description["allOf"][0]["$ref"].split("/")[-1]
    else:
        typeinfo = "No type information"
    typeinfo = f"({typeinfo})"

    msg: str = f"{param} {typeinfo}"
    default: str
    if "default" in param_description:
        default = param_description["default"]
        if default == "":
            default = "<Empty String> - May be populated by validator"
        msg = f"{msg} - Default: {default}"

    description: str
    if "description" in param_description:
        description = param_description["description"]
    else:
        description = "Unknown description."

    msg = f"{msg}\n\t{description}"
    if validators is not None:
        msg = f"{msg}\n\tValidators:"
        for validator in validators:
            if hasattr(validator, "func") and hasattr(validator.func, "__name__"):
                msg = f"{msg}\n\t\t- {validator.func.__name__}"
            else:
                msg = f"{msg}\n\t\t- <CANNOT PARSE NAME>"
    msg = f"{msg}\n\n"
    return msg


def _print_env_config(env_var: Optional[str] = None):
    print("LUTE Configuration via Environment Variables:")
    print("-" * 80)
    if env_var is not None:
        if env_var in LUTE_ENV_VARS:
            env_var_defn: EnvVarDefinition = LUTE_ENV_VARS[env_var]
            print(env_var)
            val: str
            if env_var_defn["value"] is not None:
                val = env_var_defn["value"]
            else:
                val = env_var_defn["default"]
            print(f"{env_var} = {val}")
            print(env_var_defn["description"])
    else:
        for env_var, env_var_defn in LUTE_ENV_VARS.items():
            if env_var_defn["value"] is not None:
                val = env_var_defn["value"]
            else:
                val = env_var_defn["default"]
            print(f"{env_var} = {val}")


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="Task parameters help utility.",
        description="Display parameter descriptions and types for a specified Task.",
        epilog="Refer to https://github.com/slac-lcls/lute for more information.",
    )
    parser.add_argument(
        "-e",
        "--env",
        nargs="?",
        const=True,
        default=False,
        help=(
            "Print configuration coming from env vars.\n"
            "If passed as a flag, print variable/value pairs. (lute_help -e)\n"
            "If passed with a variable, print the long description. (lute_help -e <var>)."
        ),
    )
    parser.add_argument("-l", "--list", action="store_true", help="List out all Tasks")
    parser.add_argument(
        "-T", "--Task", type=str, help="Name of the Task to inspect.", required=False
    )
    parser.add_argument(
        "--full_schema",
        action="store_true",
        help="Dump an unformated full model schema. Has more information.",
    )
    args: argparse.Namespace = parser.parse_args()

    if args.env:
        env_var: Optional[str] = args.env if isinstance(args.env, str) else None
        _print_env_config(env_var=env_var)

    task_name: str
    parameter_schema: ModelSchema
    if args.list:
        logger.info("Fetching Task list.")
        # Construct Task <-> Executor mapping
        # [task_name, [managed_task1, managed_task2, ...]
        managed_task_map: Dict[str, List[str]] = {}
        for key in dir(managed_tasks):
            obj: Any = getattr(managed_tasks, key)
            if isinstance(obj, BaseExecutor):
                task_name = obj._analysis_desc.task_result.task_name
                if task_name in managed_task_map:
                    managed_task_map[task_name].append(key)
                else:
                    managed_task_map[task_name] = [key]
        task_list_msg: str = "Task List"
        task_list_msg = f"{task_list_msg}\n{'-'*len(task_list_msg)}"
        for key in dir(lute.io.models):
            if "Parameters" in key and key not in (
                "ThirdPartyParameters",
                "TaskParameters",
                "TemplateParameters",
            ):
                task_name = key.replace("Parameters", "")
                task_list_msg = f"{task_list_msg}\n\n- {task_name}\n"
                task_list_msg = f"{task_list_msg}  Current Managed Tasks:"
                if task_name in managed_task_map:
                    task_list_msg = (
                        f"{task_list_msg} {', '.join(managed_task_map[task_name])}"
                    )
                params_obj: TaskParameters = getattr(lute.io.models, key)
                parameter_schema = cast(ModelSchema, params_obj.schema())
                if "description" in parameter_schema:
                    description: str = parameter_schema["description"]
                    for line in description.split("\n"):
                        new_line: str
                        if line:
                            new_line = f"\n\t{line}"
                        else:
                            new_line = f"\n{line}"
                        task_list_msg = f"{task_list_msg}{new_line}"

        print(task_list_msg)

    task_name = args.Task
    if task_name:
        model_name: str = f"{task_name}Parameters"

        if hasattr(lute.io.models, model_name):
            parameter_model: TaskParameters = getattr(lute.io.models, model_name)
            logger.info(f"Fetching parameter information for {task_name}.")
        else:
            logger.info(f"No Task named {task_name} found! Exiting!")
            sys.exit(-1)

        # For types need to check for key `type` or a list of dicts `anyOf=[{'type': ...}, {'type': ...}]`
        parameter_schema = cast(ModelSchema, parameter_model.schema())
        if args.full_schema:
            pprint.pprint(parameter_schema)
            sys.exit(0)

        task_description: str = parameter_schema["description"]
        required_parameters: Optional[List[Tuple[str, PropertyDict]]] = None
        if (
            "required" in parameter_schema.keys()
            and parameter_schema["required"] is not None
        ):
            required_parameters = [
                (param, parameter_schema["properties"][param])
                for param in parameter_schema["required"]
            ]

        validators: Optional[List[Callable]] = None
        out_msg: str = f"{task_name}\n{'-'*len(task_name)}\n"
        out_msg = f"{out_msg}{task_description}\n\n\n"
        if required_parameters is not None:
            out_msg = f"{out_msg}Required Parameters:\n--------------------\n"
            for param in required_parameters:
                # Ignoring mypy since the typing seems wrong. This is a list | None
                validators = (
                    parameter_model.__validators__[param[0]]  # type: ignore
                    if param[0] in parameter_model.__validators__
                    else None
                )
                out_msg = (
                    f"{out_msg}{_format_parameter_row(param[0], param[1], validators)}"
                )
            out_msg = f"{out_msg}\n\n"

        out_msg = f"{out_msg}All Parameters:\n---------------\n"
        pname: str
        for pname in parameter_schema["properties"]:
            # Ignoring mypy since the typing seems wrong. This is a list | None
            validators = (
                parameter_model.__validators__[pname]  # type: ignore
                if pname in parameter_model.__validators__
                else None
            )
            if pname == "Config":
                continue
            out_msg = f"{out_msg}{_format_parameter_row(pname, parameter_schema['properties'][pname], validators)}"

        if "definitions" in parameter_schema and parameter_schema["definitions"]:
            definitions: List[str] = [
                defn
                for defn in parameter_schema["definitions"]
                if defn not in ("AnalysisHeader", "TemplateConfig", "Config")
            ]
            if len(definitions) > 0:
                out_msg = f"{out_msg}Template Parameters:\n--------------------\n"
                for defn in definitions:
                    out_msg = f"{out_msg}{defn}:\n"
                    for pname in parameter_schema["definitions"][defn]["properties"]:
                        row: str = _format_parameter_row(
                            pname,
                            parameter_schema["definitions"][defn]["properties"][pname],
                        )
                        out_msg = f"{out_msg}{row}"
        print(out_msg)


if __name__ == "__main__":
    main()
