"""Base classes and functions for handling `Task` execution.

Executors run a `Task` as a subprocess and handle all communication with other
services, e.g., the eLog. They accept specific handlers to override default
stream parsing.

Event handlers/hooks are implemented as standalone functions which can be added
to an Executor.


Classes:
    BaseExecutor: Abstract base class from which all Executors are derived.

    Executor: Default Executor implementing all basic functionality and IPC.

    MPIExecutor: Runs exactly as the Executor but submits the Task using MPI.

Exceptions
----------

"""

__all__ = ["BaseExecutor", "Executor", "MPIExecutor"]
__author__ = "Gabriel Dorlhiac"

import copy
import logging
import os
import re
import requests
import shutil
import signal
import subprocess
import sys
import time
import warnings
from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    Union,
    cast,
    overload,
)
from typing_extensions import TypedDict, TypeAlias

from urllib3.util import Retry

from lute.execution.logging import get_logger
from lute.execution.ipc import (
    Communicator,
    LUTE_SIGNALS,
    Message,
    Party,
    PipeCommunicator,
    SocketCommunicator,
    TaskMetadataMessage,
    TaskRequest,
    TaskRequestMessage,
)
from lute.tasks.dataclasses import (
    DescribedAnalysis,
    TaskResult,
    TaskStatus,
    ElogSummaryPlots,
    TaskParametersDBReference,
)
from lute.io.parameters import RowIds, construct_task_parameters
from lute.io.models.base import (
    TaskParameters,
    TemplateParameters,
    AnalysisHeader,  # noqa: F401
    TemplateConfig,  # noqa: F401
    ThirdPartyParameters,
)  # NOTE: All imports required for unpickling!
from lute.io.db import (
    record_analysis_db,
    update_analysis_db,
    get_task_parameters_defn_and_params,
)
from lute.io.elog import post_elog_run_status, post_elog_run_table

if __debug__:
    warnings.simplefilter("default")
    os.environ["PYTHONWARNINGS"] = "default"
    logging.basicConfig(level=logging.DEBUG)
    logging.captureWarnings(True)
else:
    logging.basicConfig(level=logging.INFO)
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

logger: logging.Logger = get_logger(__name__, is_task=False)


class TaskletDict(TypedDict):
    before: Optional[List[Tuple[Callable[[Any], Any], List[Any], bool, bool]]]
    after: Optional[List[Tuple[Callable[[Any], Any], List[Any], bool, bool]]]


Executor_T: TypeAlias = "BaseExecutor"


class Hook(Protocol):
    def __call__(
        self,
        executor: Executor_T,
        msg: Message,
        proc: Optional[subprocess.Popen] = None,
    ) -> Optional[bool]: ...


class ExecutorHooks:
    """A container class for the Executor's event hooks.

    There is a corresponding function (hook) for each event/signal. Each
    function takes three parameters - a reference to the Executor, a reference
    to the Message (msg) which includes the corresponding signal and the Task
    subprocess.
    """

    __slots__ = LUTE_SIGNALS

    no_pickle_mode: Hook
    task_started: Hook
    task_failed: Hook
    task_stopped: Hook
    task_done: Hook
    task_cancelled: Hook
    task_result: Hook
    task_log: Hook
    task_request: Hook


class BaseExecutor(ABC):
    """ABC to manage Task execution and communication with user services.

    When running in a workflow, "tasks" (not the class instances) are submitted
    as `Executors`. The Executor manages environment setup, the actual Task
    submission, and communication regarding Task results and status with third
    party services like the eLog.

    Attributes:

    Methods:
        add_hook(event: str, hook: Callable[[None], None]) -> None: Create a
            new hook to be called each time a specific event occurs.

        add_default_hooks() -> None: Populate the event hooks with the default
            functions.

        update_environment(env: Dict[str, str], update_path: str): Update the
            environment that is passed to the Task subprocess.

        shell_source(env: str): Source a shell script at `env` to update the
            environment that is passed to the Task subprocess.

        execute_task(): Run the task as a subprocess.
    """

    Hooks: ClassVar[Type[ExecutorHooks]] = ExecutorHooks

    def __init__(
        self,
        task_name: str,
        communicators: List[Communicator],
        poll_interval: float = 0.05,
    ) -> None:
        """The Executor will manage the subprocess in which `task_name` is run.

        Args:
            task_name (str): The name of the Task to be submitted. Must match
                the Task's class name exactly. The parameter specification must
                also be in a properly named model to be identified.

            communicators (List[Communicator]): A list of one or more
                communicators which manage information flow to/from the Task.
                Subclasses may have different defaults, and new functionality
                can be introduced by composing Executors with communicators.

            poll_interval (float): Time to wait between reading/writing to the
                managed subprocess. In seconds.
        """
        result: TaskResult = TaskResult(
            task_name=task_name, task_status=TaskStatus.PENDING, summary="", payload=""
        )
        task_parameters: Optional[TaskParameters] = None
        task_env: Dict[str, str] = os.environ.copy()
        self._py_executable: str = sys.executable
        self._communicators: List[Communicator] = communicators

        self._analysis_desc: DescribedAnalysis = DescribedAnalysis(
            task_result=result,
            task_parameters=task_parameters,
            task_env=task_env,
            executor_name=self.__class__.__name__,
            poll_interval=poll_interval,
            communicator_desc=[""],  # Will update in _pre_task, see method for why
        )
        self._tasklets: TaskletDict = {"before": None, "after": None}
        self._shell_source_script: Optional[str] = None
        self._task_timeout: Optional[int] = None
        self._task_time0: Optional[float] = None
        self._row_ids: Optional[RowIds] = None
        self._delayed_update_env_args: Optional[
            Tuple[Union[Dict[str, str], Callable[[], Dict[str, str]]], str]
        ] = None
        self._m_task_name: str = ""

        # Check to see if we are running from the Slurm WF manager
        # It passes us a URL for status updates
        self._lute_manager_url: Optional[str] = os.getenv("LUTE_MANAGER_URL")

        #
        self._lute_maestro_session: Optional[requests.Session] = None
        if self._lute_manager_url is not None:
            self._lute_maestro_session = requests.Session()
            retries: Retry = Retry(
                total=5,
                backoff_factor=0.1,
                status_forcelist=[502, 503, 504],
                allowed_methods=["GET", "POST"],
            )
            adapter = requests.adapters.HTTPAdapter(max_retries=retries)
            self._lute_maestro_session.mount("http://", adapter)

        # We also will send the manager a jobid in addition to our name.
        # If using SLURM this is the SLURM_JOBID
        # If using straight Python launching the process ID
        self._job_id: str = os.getenv("SLURM_JOBID", str(os.getpid()))

        self._maestro_heartbeat_s: float = float(
            os.getenv("LUTE_MAESTRO_HEARTBEAT", 30.0)
        )
        # We also update task_last_update if a message of ANY kind is
        # received -- This is done in _task_loop (if overriden, must be done
        # by overriding class as well). This is used as proxy of Task life.
        # Maestro uses the env var LUTE_NO_HEARTBEAT_KILL to consider the
        # unresponsive timeout. We'll do the same here. If last update
        # was longer than LUTE_NO_HEARTBEAT_KILL or a default of 900.0 seconds
        # ago. Consider it UNRESPONSIVE and dead.
        self._task_last_update: float = 0.0  # Irrespective of timeout, track updates
        self._no_heartbeat_kill: float = float(
            os.getenv("LUTE_NO_HEARTBEAT_KILL", 900.0)
        )

    @property
    def task_name(self) -> str:
        return self._analysis_desc.task_result.task_name

    @task_name.setter
    def task_name(self, new_name: str) -> None:
        self._analysis_desc.task_result.task_name = new_name

    @property
    def managed_task_name(self) -> str:
        return self._m_task_name

    def _report_to_manager(
        self,
        end_point: str,
        json_data: Optional[Dict[str, str]] = None,
        method: str = "POST",
    ) -> Any:
        try:
            if self._lute_maestro_session is not None:
                func = getattr(self._lute_maestro_session, method.lower())
            else:
                func = getattr(requests, method.lower())
        except AttributeError:
            logger.error(f"Unable to send an HTTP request of type {method}")
            return

        # Include a job id here so it doesn't need to be remembered at each call site
        tagged_json_data: Dict[str, str] = {"orig_job_id": self._job_id}
        if json_data is not None:
            tagged_json_data.update(json_data)

        # Set a timeout so we don't hang if the workflow manager dies
        timeout: float = 5.0
        try:
            resp: requests.models.Response
            if json_data is not None:
                resp = func(
                    f"http://{self._lute_manager_url}/{end_point}",
                    json=tagged_json_data,
                    timeout=timeout,
                )
            else:
                resp = func(
                    f"http://{self._lute_manager_url}/{end_point}", timeout=timeout
                )

            if hasattr(resp, "json"):
                try:
                    good_json: Any = resp.json()
                    return good_json
                except requests.JSONDecodeError:
                    # logger.debug("Bad json.")
                    # Don't know if we want to check this?
                    # Probably available via headers whether it should be decoded
                    # as json
                    ...
            if hasattr(resp, "content"):
                return resp.content
        except requests.ConnectTimeout:
            logger.error(
                f"HTTP request to workflow manager timed out after {timeout} seconds!"
            )
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection Failed: {e}")

    def add_tasklet(
        self,
        tasklet: Callable,
        args: List[Any],
        when: Union[Literal["before"], Literal["after"]] = "after",
        set_result: bool = False,
        set_summary: bool = False,
    ) -> None:
        """Add/register a tasklet to be run by the Executor.

        Adds a tasklet to be run by the Executor in addition to the main Task.
        The tasklet can be run before or after the main Task has been run.

        Args:
            tasklet (Callable[[Any], Any]): The tasklet (function) to run.

            args (List[Any]): A list of all the arguments to be passed to the
                tasklet. Arguments can include substitutions for parameters to
                be extracted from the TaskParameters object. The same jinja-like
                syntax used in configuration file substiutions is used to specify
                a parameter substitution in an argument. E.g. if a Task to be
                run has a parameter `input_file`, the parameter can be substituted
                in the tasklet argument using: `"{{ input_file  }}"`. Note that
                substitutions must be passed as strings. Conversions will be done
                during substitution if required.

            when (str): When to run the tasklet. Either `before` or `after` the
                main Task. Default is after.

            set_result (bool): Whether to use the output from the tasklet as the
                result of the main Task. Default is False.

            set_summary (bool): Whether to use the output from the tasklet as the
                summary of the main Task. Default is False.
        """
        if when not in ("before", "after"):
            logger.error("Can only run tasklet `before` or `after` Task! Ignoring...")
            return
        tasklet_tuple: Tuple[Callable[[Any], Any], List[Any], bool, bool]
        tasklet_tuple = (tasklet, args, set_result, set_summary)
        if self._tasklets[when] is None:
            self._tasklets[when] = [tasklet_tuple]
        else:
            assert isinstance(self._tasklets[when], list)
            cast(list, self._tasklets[when]).append(tasklet_tuple)

    def _sub_tasklet_parameters(self, args: List[Any]) -> List[Any]:
        """Substitute tasklet arguments using TaskParameters members."""
        sub_pattern = r"\{\{[^}{]*\}\}"
        new_args: List[Any] = []
        for arg in args:
            new_arg: Any = arg
            matches: List[str] = re.findall(sub_pattern, arg)
            for m in matches:
                param_name: str = m[2:-2].strip()  # Remove {{}}
                params: Any = self._analysis_desc.task_parameters
                if "." in param_name:
                    # Iterate so we can substitute e.g. {{ lute_config.run }}
                    hier_param_list: List[str] = param_name.split(".")
                    for idx, param in enumerate(hier_param_list):
                        if hasattr(params, param):
                            if idx != len(hier_param_list) - 1:
                                params = getattr(params, param)
                            param_name = param
                        else:
                            break
                if hasattr(params, param_name):
                    pattern: str = m.replace("{{", r"\{\{").replace("}}", r"\}\}")
                    sub: Any = getattr(params, param_name)
                    new_arg = re.sub(pattern, str(sub), new_arg)
                if new_arg.isnumeric():
                    new_arg = int(new_arg)
                else:
                    try:
                        new_arg = float(new_arg)
                    except ValueError:
                        pass
            new_args.append(new_arg)
        return new_args

    def _run_tasklets(
        self, *, when: Union[Literal["before"], Literal["after"]]
    ) -> None:
        """Run all tasklets of the specified kind."""
        if when not in self._tasklets.keys():
            logger.error(f"Ignore request to run tasklets of unknown kind: {when}")
            return
        if self._tasklets[when] is None:
            logger.debug(f"No tasklets to run {when}.")
            return
        for tasklet_spec in cast(list, self._tasklets[when]):
            tasklet: Callable[[Any], Any]
            args: List[Any]
            set_result: bool
            set_summary: bool
            tasklet, args, set_result, set_summary = tasklet_spec
            args = self._sub_tasklet_parameters(args)
            logger.debug(f"Running {tasklet} with {args}")
            output: Any
            try:
                output = tasklet(*args)  # Many don't return anything
            except Exception as err:
                logger.error(f"Tasklet failed! Error: {err}")
                output = None
            # We set result payloads or summaries now, but the processing is done
            # by process_results method called sometime after the last tasklet
            tmp: Any
            if set_result and output is not None:
                if isinstance(self._analysis_desc.task_result.payload, list):
                    # We have multiple payloads to process, append to list
                    self._analysis_desc.task_result.payload.append(output)
                elif self._analysis_desc.task_result.payload != "":
                    # We have one payload already, convert to list and append
                    tmp = self._analysis_desc.task_result.payload
                    self._analysis_desc.task_result.payload = []
                    self._analysis_desc.task_result.payload.append(tmp)
                    self._analysis_desc.task_result.payload.append(output)
                else:
                    # Payload == "" - i.e. hasn't been set
                    self._analysis_desc.task_result.payload = output
            if set_summary and output is not None:
                if isinstance(self._analysis_desc.task_result.summary, list):
                    # We have multiple summary objects to process, append to list
                    self._analysis_desc.task_result.summary.append(output)
                elif self._analysis_desc.task_result.summary != "":
                    # We have one summary already, convert to list and append
                    tmp = self._analysis_desc.task_result.summary
                    self._analysis_desc.task_result.summary = []
                    self._analysis_desc.task_result.summary.append(tmp)
                    self._analysis_desc.task_result.summary.append(output)
                else:
                    # Summary == "" - i.e. hasn't been set
                    self._analysis_desc.task_result.summary = output

    def add_hook(
        self,
        event: str,
        # hook: Callable[[Self, Message, Optional[subprocess.Popen]], Optional[bool]],
        hook: Hook,
    ) -> None:
        """Add a new hook.

        Each hook is a function called any time the Executor receives a signal
        for a particular event, e.g. Task starts, Task ends, etc. Calling this
        method will remove any hook that currently exists for the event. I.e.
        only one hook can be called per event at a time. Creating hooks for
        events which do not exist is not allowed.

        Args:
            event (str): The event for which the hook will be called.

            hook (Callable[[None], None]) The function to be called during each
                occurrence of the event.
        """
        if event.upper() in LUTE_SIGNALS:
            setattr(self.Hooks, event.lower(), hook)

    @abstractmethod
    def add_default_hooks(self) -> None:
        """Populate the set of default event hooks."""

        ...

    @overload
    def update_environment(
        self, env: Dict[str, str], update_path: str = "prepend"
    ) -> None:
        """Update the stored set of environment variables.

        These are passed to the subprocess to setup its environment.

        Args:
            env (Dict[str, str]): A dictionary of "VAR":"VALUE" pairs of
                environment variables to be added to the subprocess environment.
                If any variables already exist, the new variables will
                overwrite them (except PATH and PYTHONPATH, see below).

            update_path (str): If PATH and/or PYTHONPATH are present in the new
                set of variables, this argument determines how the old value is
                dealt with. There are three options:
                * "prepend" : The new values are prepended to the old ones.
                * "append" : The new values are appended to the old ones.
                * "overwrite" : The old value is overwritten by the new one.
                "prepend" is the default option. If PATH and/or PYTHONPATH is not
                present in the current environment, the new PATH is used without
                modification.
        """
        ...

    @overload
    def update_environment(self, env: Callable[[], Dict[str, str]]) -> None:
        """Update the stored set of environment variables.

        These are passed to the subprocess to setup its environment.

        Args:
            env (Callable[[],Dict[str, str]]): A managed-Task specific function
                which returns a dictionary of environment variables to include
                in the Task environment. This function can implement more complex
                logic to determine values for the specific environment variables.
        """
        ...

    def update_environment(
        self,
        env: Union[Dict[str, str], Callable[[], Dict[str, str]]],
        update_path: str = "prepend",
    ) -> None:
        """Update the stored set of environment variables.

        These are passed to the subprocess to setup its environment.

        Args:
            env (Union[Dict[str, str], Callable[[],Dict[str, str]]]): If a dictionary,
                it contains a series of "VAR":"VALUE" pairs of environment variables to
                be added to the subprocess environment. If any variables already exist,
                the new variables will overwrite them (except PATH/PYTHONPATH, see below).
                If a callable, a managed-Task specific function which returns a dictionary
                of environment variables to include in the Task environment. This function
                can implement more complex logic to determine values for the specific
                environment variables. If it is a callable, the `update_path` argument
                to this method is ignored.

            update_path (str): If PATH and/or PYTHONPATH is present in the new
                set of variables, this argument determines how the old value is
                dealt with. There are three options:
                * "prepend" : The new PATH values are prepended to the old ones.
                * "append" : The new PATH values are appended to the old ones.
                * "overwrite" : The old PATH is overwritten by the new one.
                "prepend" is the default option. If PATH is not present in the
                current environment, the new PATH is used without modification.
        """
        self._delayed_update_env_args = (env, update_path)

    def _update_environment(
        self,
        env: Union[Dict[str, str], Callable[[], Dict[str, str]]],
        update_path: str = "prepend",
        use_tenv_prefix: bool = False,
    ) -> None:
        env_update: Dict[str, str]
        if callable(env):
            raw_env_update: Dict[str, str] = env()
            if use_tenv_prefix:
                env_update = {
                    f"LUTE_TENV_{key}": val for key, val in raw_env_update.items()
                }
            else:
                env_update = raw_env_update
            self._analysis_desc.task_env.update(env_update)
            return

        for key in ("PATH", "PYTHONPATH"):
            if key in env and key in self._analysis_desc.task_env:
                sep: str = os.pathsep
                if update_path == "prepend":
                    env[key] = f"{env[key]}{sep}{self._analysis_desc.task_env[key]}"
                elif update_path == "append":
                    env[key] = f"{self._analysis_desc.task_env[key]}{sep}{env[key]}"
                elif update_path == "overwrite":
                    pass
                else:
                    raise ValueError(
                        (
                            f"{update_path} is not a valid option for `update_path`!"
                            " Options are: prepend, append, overwrite."
                        )
                    )
        if use_tenv_prefix:
            env_update = {f"LUTE_TENV_{key}": val for key, val in env.items()}
        else:
            env_update = env
        self._analysis_desc.task_env.update(env_update)

    def shell_source(self, env: str) -> None:
        """Source a script.

        Unlike `update_environment` this method sources a new file.

        We prepend a token to each environment variable. This allows the initial
        part of the Task to be run using the appropriate environment.

        The environment variables containing the token will be swapped in using
        their appropriate form prior to the actual execution of Task code.

        Args:
            env (str): Path to the script to source.
        """
        self._shell_source_script = env

    def _shell_source(self) -> None:
        """Actually shell source step.

        This is run prior to Task execution.
        """
        if self._shell_source_script is None:
            logger.error("Called _shell_source without defining source script!")
            return
        if not os.path.exists(self._shell_source_script):
            logger.error(f"Cannot source environment from {self._shell_source_script}!")
            return
        # Get both the environment and the python version of the target environment
        script: str = (
            f"set -a\n"
            f'source "{self._shell_source_script}" >/dev/null\n'
            f'NEW_PYVER=$(python3 -c "import sys; '
            "print(f'python{sys.version_info.major}.{sys.version_info.minor}')\")\n"
            f'python3 -c "import os; env=dict(os.environ); '
            "env['LUTE_NEW_PYVER']=os.environ.get('NEW_PYVER', ''); "
            'print(env)"\n'
        )
        logger.info(f"Sourcing file {self._shell_source_script}")
        subproc_env: Dict[str, str] = {}
        lute_path: Optional[str] = os.getenv("LUTE_PATH")
        lute_virtual_env: Optional[str] = os.getenv("LUTE_VIRTUAL_ENV")
        for key, val in os.environ.items():
            if lute_virtual_env is not None:
                if "CONDA" not in key and "PYTHONPATH" not in key:
                    subproc_env[key] = val
            else:
                if "CONDA" not in key:
                    if key == "PYTHONPATH":
                        # Strip out psana2 environment leakage
                        curr_parts: List[str] = val.split(":")
                        cleaned_parts: List[str] = []
                        for part in curr_parts:
                            part_lower: str = part.lower()
                            is_lute: bool = "lute" in part_lower
                            if lute_path:
                                is_lute = (
                                    is_lute
                                    or (lute_path.lower() in part_lower)
                                    or (part_lower in lute_path.lower())
                                )
                            if is_lute:
                                cleaned_parts.append(part)
                        subproc_env[key] = ":".join(cleaned_parts)
                    else:
                        subproc_env[key] = val

        o, e = subprocess.Popen(
            ["bash", "-c", script], stdout=subprocess.PIPE, env=subproc_env
        ).communicate()
        tmp_environment: Dict[str, str] = eval(o)
        new_environment: Dict[str, str] = {}

        # For picking up LUTE, the new environment may be a different python version
        # So we need to make sure to pick it up appropriately for C-extension usage
        new_pyver: str = tmp_environment.get("LUTE_NEW_PYVER", "python3.9")
        old_pyver: str = f"python{sys.version_info[0]}.{sys.version_info[1]}"
        for key, value in tmp_environment.items():
            # Make sure LUTE vars are available
            if "LUTE_" in key or "SLURM_" in key or key in ("RUN", "EXPERIMENT"):
                new_environment[key] = value
            else:
                new_environment[f"LUTE_TENV_{key}"] = value
                # Make sure we don't accidentally lose our current environment
                if key in os.environ and os.getenv(key) == value:
                    # Add identical items first
                    new_environment[key] = value
                elif key in ("PYTHONPATH", "PATH"):
                    curr: Optional[str] = os.getenv(key)
                    if curr is not None:
                        if curr in value:
                            new_environment[key] = f"{curr}:{value}"
                        else:
                            # Make sure to keep our env first, for dependencies
                            # e.g. pydantic
                            new_environment[key] = f"{curr}:{value}"
                            # For the TENV, make sure they get the environment requested
                            if key == "PATH":
                                new_environment[f"LUTE_TENV_{key}"] = f"{value}:{curr}"
                            else:
                                new_environment[f"LUTE_TENV_{key}"] = value

        # Until we make LUTE installable... Need to make sure this is available
        # for first-party Tasks, regardless of the directory they run in if using
        # a new environment
        old_python_path: str = new_environment.get("PYTHONPATH", "")
        new_python_path: str = new_environment.get("LUTE_TENV_PYTHONPATH", "")
        new_lute_path: Optional[str] = lute_path
        if lute_path is None:
            logger.warning("LUTE_PATH not defined! Task may fail to find LUTE!")

        elif lute_virtual_env is None:
            assert new_lute_path
            if old_python_path:
                new_environment["PYTHONPATH"] = f"{lute_path}:{old_python_path}"
            else:
                new_environment["PYTHONPATH"] = lute_path

            if new_pyver not in lute_path:
                # We have a new lute_path to use for a different Python version
                new_lute_path = lute_path.replace(old_pyver, new_pyver)
                logger.debug(f"Task will use LUTE from: {new_lute_path}")

            if not os.path.exists(new_lute_path):
                logger.warning(
                    f"Task will be running in {new_pyver}, but no LUTE installation "
                    "for that version exists! Things may fail if depending on C "
                    "extensions!"
                )
            else:
                # Because of the venv updates below, setting PYTHONPATH isn't
                # sufficient to pickup new Python. Need to make sure LUTE_PATH is set
                new_environment["LUTE_TENV_LUTE_PATH"] = new_lute_path

            if new_python_path:
                new_environment["LUTE_TENV_PYTHONPATH"] = (
                    f"{new_lute_path}:{new_python_path}"
                )
            elif new_lute_path:
                new_environment["LUTE_TENV_PYTHONPATH"] = new_lute_path

        # Update python executable and lute path to correct version
        elif lute_virtual_env is not None:
            if new_pyver != old_pyver:
                # Strip new_pyver version
                version: str = new_pyver.replace("python", "").replace(".", "")
                new_py_executable = os.getenv(f"LUTE_VIRTUAL_ENV_PY{version}")
                if new_py_executable is not None:
                    self._py_executable = new_py_executable
                    lute_env_root = os.path.dirname(os.path.dirname(new_py_executable))
                    new_lute_path = os.path.join(
                        lute_env_root, "lib", new_pyver, "site-packages"
                    )
                    new_environment["LUTE_TENV_LUTE_PATH"] = new_lute_path
                else:
                    logger.warning(
                        f"Task needs to run in {new_pyver}, but no {version} "
                        "installation is available! Task may fail."
                    )

        self._analysis_desc.task_env = new_environment

    def _pre_task(self) -> str:
        """Any actions to be performed before task submission.

        This method should be modified carefully, if at all, by subclasses as
        it prepares environments. This preparation is rather finicky given the
        need to prevent collisions between various environments.

        Returns:
            lute_path (str): The path to the LUTE installation being used.
        """
        # This prevents the Executors in managed_tasks.py from all acquiring
        # resources like sockets.
        # Some Communicators setup descriptions during delayed_setup
        # so we update our analysis description now.
        communicator_desc: List[str] = []
        for communicator in self._communicators:
            communicator.delayed_setup()
            # Not great, but experience shows we need a bit of time to setup
            # network.
            time.sleep(0.1)
            communicator.stage_communicator()
            communicator_desc.append(str(communicator))
        self._analysis_desc.communicator_desc = communicator_desc
        # Propagate any env vars setup by Communicators - only update LUTE_ vars
        tmp: Dict[str, str] = {
            key: os.environ[key] for key in os.environ if "LUTE_" in key
        }
        self._analysis_desc.task_env.update(tmp)

        # ********* Important ********* #
        # If using _update_environment AND _shell_source, the environment
        # variables in _update_environment must be prepended by LUTE_TENV_
        lute_path: Optional[str] = os.getenv("LUTE_PATH")
        if lute_path is None:
            logger.debug("Absolute path to subprocess_task.py not found.")
            lute_path = os.path.abspath(f"{os.path.dirname(__file__)}/../..")
            os.environ.update({"LUTE_PATH": lute_path})
            self._analysis_desc.task_env.update({"LUTE_PATH": lute_path})

        use_tenv_prefix: bool = False
        if self._shell_source_script is not None:
            self._shell_source()
            use_tenv_prefix = True

        if self._delayed_update_env_args is not None:
            self._update_environment(
                *self._delayed_update_env_args, use_tenv_prefix=use_tenv_prefix
            )

        return lute_path

    def _submit_task(self, cmd: str) -> subprocess.Popen:
        proc: subprocess.Popen = subprocess.Popen(
            cmd.split(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self._analysis_desc.task_env,
        )
        if proc.stdin is not None:
            os.set_blocking(proc.stdin.fileno(), False)
        if proc.stdout is not None:
            os.set_blocking(proc.stdout.fileno(), False)
        if proc.stderr is not None:
            os.set_blocking(proc.stderr.fileno(), False)
        return proc

    @abstractmethod
    def _task_loop(self, proc: subprocess.Popen) -> None:
        """Actions to perform while the Task is running.

        This function is run in the body of a loop until the Task signals
        that its finished.
        """
        ...

    @abstractmethod
    def _finalize_task(self, proc: subprocess.Popen) -> None:
        """Any actions to be performed after the Task has ended.

        Examples include a final clearing of the pipes, retrieving results,
        reporting to third party services, etc.
        """
        ...

    def _submit_cmd(self, executable_path: str, params: str) -> str:
        """Return a formatted command for launching Task subprocess.

        May be overridden by subclasses.

        The default submission uses the Executor environment. This ensures that
        all necessary packages (e.g. Pydantic for validation) are available to
        the startup scripts. If a Task has a different environment it will be
        swapped prior to execution.

        Args:
            executable_path (str): Path to the LUTE subprocess script.

            params (str): String of formatted command-line arguments.

        Returns:
            cmd (str): Appropriately formatted command for this Executor.
        """
        cmd: str = ""
        if __debug__:
            cmd = f"{self._py_executable} -B {executable_path} {params}"
        else:
            cmd = f"{self._py_executable} -OB {executable_path} {params}"

        return cmd

    def execute_task(self) -> None:
        """Run the requested Task as a subprocess."""
        # _pre_task does Task environment preparation. All updates are done now
        # to prevent various Managed Tasks which are all defined in the same module
        # from affecting each other.
        lute_path: str = self._pre_task()
        executable_path: Optional[str] = shutil.which("subprocess_task")
        if executable_path is None:
            # Did not install and just running from a clone of repo
            executable_path = f"{lute_path}/subprocess_task.py"
        config_path: str = self._analysis_desc.task_env["LUTE_CONFIGPATH"]
        params: str = f"-c {config_path} -t {self._analysis_desc.task_result.task_name}"

        cmd: str = self._submit_cmd(executable_path, params)
        proc: subprocess.Popen = self._submit_task(cmd)
        self._task_time0 = time.monotonic()
        # Count silence from submission, not from 0.0: monotonic() is time since boot, so a
        # 0.0 start made every Task that was quiet for its first heartbeat look 900 s dead.
        self._task_last_update = self._task_time0
        # In the event we were using generated parameters, we may have a _XX suffix
        # Now that the Task has been submitted, we can remove that from the name
        # for storage in the database - just reset the name
        self.task_name = re.sub(r"_\d+$", "", self.task_name)

        if self._lute_manager_url is not None:
            # On STARTED we will store some information that could be useful for `maestro`
            json_data: Dict[str, str] = {
                "managed_task": self._m_task_name,
                "task": self.task_name,
                "status": "STARTED",
                # This should be set by now
                "executor_hostname": os.getenv("LUTE_EXECUTOR_HOST", "UNKNOWN"),
            }
            self._report_to_manager(end_point="status", json_data=json_data)

        # Send periodic heartbeats for liveness tracking
        last_heartbeat: float = time.monotonic()

        affinity: Set[int] = os.sched_getaffinity(0)
        # By convention, the Executor takes the minimum core on this node.
        # Task gets everything else. If we only have 1 core here then out of luck
        # and cannot set new affinities without issues
        if len(affinity) > 1:
            executor_affinity: Set[int] = {min(affinity)}
            os.sched_setaffinity(0, executor_affinity)
        while self._task_is_running(proc):
            self._task_loop(proc)
            if self._task_timeout is not None:
                run_time: float = time.monotonic() - self._task_time0
                if run_time > self._task_timeout:
                    logger.error("Task timed out!")
                    self._sigalrm_task(proc)

            now: float = time.monotonic()
            if (now - last_heartbeat) > self._maestro_heartbeat_s:
                last_heartbeat = now
                # If unresponsive for longer than kill period, we exit.
                # Otherwise, just send the normal heartbeat ping - message doesn't
                # matter. Just say RUNNING
                stat_str: str = "RUNNING"
                if (now - self._task_last_update) > self._no_heartbeat_kill:
                    stat_str = "UNRESPONSIVE"

                # Break early - the Executor will send a message at the very end
                # `maestro` is faster than Python. We want it to wait around until
                # the last minute, so we only send the UNRESPONSIVE update before tear-down
                if stat_str == "UNRESPONSIVE":
                    unresp_msg: str = (
                        f"Task did not respond for more than {self._no_heartbeat_kill} seconds. "
                        "It is marked as UNRESPONSIVE and failed!"
                    )
                    logger.error(unresp_msg)
                    if self._lute_manager_url is not None:
                        json_log_data: Dict[str, str] = {
                            "managed_task": self._m_task_name,
                            "message": unresp_msg,
                        }
                        self._report_to_manager(
                            end_point="log", json_data=json_log_data
                        )
                    self._kill_task(proc=proc, status=TaskStatus.UNRESPONSIVE)
                    break

                # Since we may run without maestro, we will check the UNRESPONSIVE
                # outside the lute_manager_url check.
                if self._lute_manager_url is not None:
                    json_data = {
                        "managed_task": self._m_task_name,
                        "task": self.task_name,
                        "status": stat_str,
                    }
                    self._report_to_manager(end_point="status", json_data=json_data)

            time.sleep(self._analysis_desc.poll_interval)

        if proc.stdin is not None:
            os.set_blocking(proc.stdin.fileno(), True)
        if proc.stdout is not None:
            os.set_blocking(proc.stdout.fileno(), True)
        if proc.stderr is not None:
            os.set_blocking(proc.stderr.fileno(), True)

        self._finalize_task(proc)
        if proc.stdin is not None:
            proc.stdin.close()
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()
        proc.wait()
        if ret := proc.returncode:
            if self._analysis_desc.task_result.task_status != TaskStatus.TIMEDOUT:
                logger.warning(f"Task failed with return code: {ret}")
                self._analysis_desc.task_result.task_status = TaskStatus.FAILED
                self.Hooks.task_failed(self, msg=Message())
        elif self._analysis_desc.task_result.task_status == TaskStatus.RUNNING:
            # Ret code is 0, no exception was thrown, task forgot to set status
            self._analysis_desc.task_result.task_status = TaskStatus.COMPLETED
            logger.debug("Task did not change from RUNNING status. Assume COMPLETED.")
            self.Hooks.task_done(self, msg=Message())
        if self._tasklets["after"] is not None:
            # Tasklets before results processing since they may create result
            self._run_tasklets(when="after")

        try:
            self.process_results()
        except Exception as err:
            logger.critical(
                f"Unable to process results! Downstream Tasks may fail! {err}"
            )

        try:
            self._store_configuration()
        except Exception as err:
            logger.critical(
                f"Unable to store configuration! Downstream Tasks may fail! {err}"
            )

        for comm in self._communicators:
            comm.clear_communicator()
        time.sleep(1)
        status: TaskStatus = self._analysis_desc.task_result.task_status
        status_str: str
        if status == TaskStatus.FAILED:
            status_str = "FAILED"
        elif status == TaskStatus.UNRESPONSIVE:
            status_str = "UNRESPONSIVE"
        elif status == TaskStatus.CANCELLED:
            status_str = "CANCELLED"
        elif status == TaskStatus.TIMEDOUT:
            status_str = "TIMEDOUT"
        else:
            status_str = "COMPLETED"

        hostfile: Optional[str] = os.getenv("LUTE_MPI_HOSTFILE_PATH")
        if hostfile is not None:
            if os.path.exists(hostfile):
                logger.debug(f"Removing (temporary) MPI hostfile: {hostfile}.")
                os.remove(hostfile)

        if self._lute_manager_url is not None:
            json_data = {
                "managed_task": self._m_task_name,
                "status": status_str,
            }
            self._report_to_manager(end_point="status", json_data=json_data)
        if self._analysis_desc.task_result.task_status in (
            TaskStatus.FAILED,
            TaskStatus.TIMEDOUT,
            TaskStatus.CANCELLED,
            TaskStatus.UNRESPONSIVE,
        ):
            logger.info("Exiting after Task failure. Result recorded.")
            logging.shutdown()
            time.sleep(0.5)  # Try to give a chance for any unsent messages to drain
            sys.exit(-1)
        logger.info("Exiting after Task completion.")
        logging.shutdown()

    def _store_configuration(self) -> None:
        """Store configuration and results in the LUTE database."""
        if self._row_ids is not None:
            update_analysis_db(
                cfg=copy.deepcopy(self._analysis_desc), row_ids=self._row_ids
            )
        else:
            record_analysis_db(cfg=copy.deepcopy(self._analysis_desc))

    def _task_is_running(self, proc: subprocess.Popen) -> bool:
        """Whether a subprocess is running.

        Args:
            proc (subprocess.Popen): The subprocess to determine the run status
                of.

        Returns:
            bool: Is the subprocess task running.
        """
        # Add additional conditions - don't want to exit main loop
        # if only stopped
        task_status: TaskStatus = self._analysis_desc.task_result.task_status
        is_running: bool = task_status != TaskStatus.COMPLETED
        is_running &= task_status != TaskStatus.CANCELLED
        is_running &= task_status != TaskStatus.TIMEDOUT
        is_running &= task_status != TaskStatus.FAILED
        return proc.poll() is None and is_running

    def _stop(self, proc: subprocess.Popen) -> None:
        """Stop the Task subprocess."""
        os.kill(proc.pid, signal.SIGTSTP)
        self._analysis_desc.task_result.task_status = TaskStatus.STOPPED

    def _sigalrm_task(self, proc: subprocess.Popen) -> None:
        """Timeout the Task subprocess with SIGALRM."""
        os.kill(proc.pid, signal.SIGALRM)
        self._analysis_desc.task_result.task_status = TaskStatus.TIMEDOUT

    def _kill_task(
        self, proc: subprocess.Popen, status: TaskStatus = TaskStatus.UNRESPONSIVE
    ) -> None:
        """Kill the Task subprocess with SIGKILL.

        Args:
            proc (subprocess.Popen): The subprocess to signal.

            status (TaskStatus): The status to mark the Task with after signalling.
                By default, this will be UNRESPONSIVE, as this is the usual reason
                that this method would be employed (as opposed to SIGALRM above). But
                a separate status can be used instead.
        """
        os.kill(proc.pid, signal.SIGKILL)
        self._analysis_desc.task_result.task_status = status

    def _continue(self, proc: subprocess.Popen) -> None:
        """Resume a stopped Task subprocess."""
        os.kill(proc.pid, signal.SIGCONT)
        self._analysis_desc.task_result.task_status = TaskStatus.RUNNING

    def _set_result_from_parameters(self) -> None:
        """Use TaskParameters object to set TaskResult fields.

        A result may be defined in terms of specific parameters. This is most
        useful for ThirdPartyTasks which would not otherwise have an easy way of
        reporting what the TaskResult is. There are two options for specifying
        results from parameters:
            1. A single parameter (Field) of the model has an attribute
               `is_result`. This is a bool indicating that this parameter points
               to a result. E.g. a parameter `output` may set `is_result=True`.
            2. The `TaskParameters.Config` has a `result_from_params` attribute.
               This is an appropriate option if the result is determinable for
               the Task, but it is not easily defined by a single parameter. The
               TaskParameters.Config.result_from_param can be set by a custom
               validator, e.g. to combine the values of multiple parameters into
               a single result. E.g. an `out_dir` and `out_file` parameter used
               together specify the result. Currently only string specifiers are
               supported.

        A TaskParameters object specifies that it contains information about the
        result by setting a single config option:
                        TaskParameters.Config.set_result=True
        In general, this method should only be called when the above condition is
        met, however, there are minimal checks in it as well.
        """
        # This method shouldn't be called unless appropriate
        # But we will add extra guards here
        if self._analysis_desc.task_parameters is None:
            logger.debug(
                "Cannot set result from TaskParameters. TaskParameters is None!"
            )
            return
        if (
            not hasattr(self._analysis_desc.task_parameters.Config, "set_result")
            or not self._analysis_desc.task_parameters.Config.set_result
        ):
            logger.debug(
                "Cannot set result from TaskParameters. `set_result` not specified!"
            )
            return

        # First try to set from result_from_params (faster)
        if hasattr(self._analysis_desc.task_parameters, "_result_from_params"):
            result_from_params: str = (
                self._analysis_desc.task_parameters._result_from_params
            )
            logger.info(f"TaskResult specified as {result_from_params}.")
            self._analysis_desc.task_result.payload = result_from_params
            del self._analysis_desc.task_parameters._result_from_params
        else:
            # Iterate parameters to find the one that is the result
            schema: Dict[str, Any] = self._analysis_desc.task_parameters.schema()
            for param, value in self._analysis_desc.task_parameters.dict().items():
                if param == "_result_from_params":
                    continue
                param_attrs: Dict[str, Any]
                if isinstance(value, TemplateParameters):
                    # Extract TemplateParameters if needed
                    value = value.params
                    extra_models: List[str] = schema["definitions"].keys()
                    for model in extra_models:
                        if model in ("AnalysisHeader", "TemplateConfig"):
                            continue
                        if param in schema["definitions"][model]["properties"]:
                            param_attrs = schema["definitions"][model]["properties"][
                                param
                            ]
                            break
                    else:
                        if isinstance(
                            self._analysis_desc.task_parameters, ThirdPartyParameters
                        ):
                            param_attrs = self._analysis_desc.task_parameters._unknown_template_params[
                                param
                            ]
                        else:
                            raise ValueError(
                                f"No parameter schema for {param}. Check model!"
                            )
                else:
                    param_attrs = schema["properties"][param]
                if "is_result" in param_attrs:
                    is_result: bool = param_attrs["is_result"]
                    if isinstance(is_result, bool) and is_result:
                        logger.info(f"TaskResult specified as {value}.")
                        self._analysis_desc.task_result.payload = value
                    else:
                        logger.debug(
                            (
                                f"{param} specified as result! But specifier is of "
                                f"wrong type: {type(is_result)}!"
                            )
                        )
                    break  # We should only have 1 result-like parameter!

        # If we get this far and haven't changed the payload we should complain
        task_name: str
        if self._analysis_desc.task_result.payload == "":
            task_name = self._analysis_desc.task_result.task_name
            logger.debug(
                (
                    f"{task_name} specified result be set from {task_name}Parameters,"
                    " but no result provided! Check model definition!"
                )
            )
        # Now check for impl_schemas and pass to result.impl_schemas
        # Currently unused
        impl_schemas: Optional[str] = (
            self._analysis_desc.task_parameters.Config.impl_schemas
        )
        self._analysis_desc.task_result.impl_schemas = impl_schemas
        # If we set_result but didn't get schema information we should complain
        if self._analysis_desc.task_result.impl_schemas is None:
            task_name = self._analysis_desc.task_result.task_name
            logger.debug(
                (
                    f"{task_name} specified result be set from {task_name}Parameters,"
                    " but no schema provided! Check model definition!"
                )
            )

    def process_results(self) -> None:
        """Perform any necessary steps to process TaskResults object.

        Processing will depend on subclass. Examples of steps include, moving
        files, converting file formats, compiling plots/figures into an HTML
        file, etc.
        """
        self._process_results()

    @abstractmethod
    def _process_results(self) -> None: ...


class Executor(BaseExecutor):
    """Basic implementation of an Executor which manages simple IPC with Task.

    Attributes:

    Methods:
        add_hook(event: str, hook: Callable[[None], None]) -> None: Create a
            new hook to be called each time a specific event occurs.

        add_default_hooks() -> None: Populate the event hooks with the default
            functions.

        update_environment(env: Dict[str, str], update_path: str): Update the
            environment that is passed to the Task subprocess.

        execute_task(): Run the task as a subprocess.
    """

    def __init__(
        self,
        task_name: str,
        communicators: List[Communicator] = [
            PipeCommunicator(Party.EXECUTOR),
            SocketCommunicator(Party.EXECUTOR),
        ],
        poll_interval: float = 0.05,
    ) -> None:
        super().__init__(
            task_name=task_name,
            communicators=communicators,
            poll_interval=poll_interval,
        )
        self.add_default_hooks()

    def add_default_hooks(self) -> None:
        """Populate the set of default event hooks."""

        def no_pickle_mode(
            executor: Executor_T,
            msg: Message,
            proc: Optional[subprocess.Popen] = None,
        ) -> Optional[bool]:
            for idx, communicator in enumerate(executor._communicators):
                if isinstance(communicator, PipeCommunicator):
                    executor._communicators[idx] = PipeCommunicator(
                        Party.EXECUTOR, use_pickle=False
                    )
            return None

        self.add_hook("no_pickle_mode", no_pickle_mode)

        def task_started(
            executor: Executor_T,
            msg: Message,
            proc: Optional[subprocess.Popen] = None,
        ) -> Optional[bool]:
            assert proc is not None
            if isinstance(msg.contents, TaskParameters):
                executor._analysis_desc.task_parameters = msg.contents
            elif isinstance(msg.contents, TaskParametersDBReference):
                work_dir: str = msg.contents.db_dir
                row_ids: RowIds = msg.contents.row_ids
                definition: Dict[str, Any]
                param_values: Dict[str, Any]
                definition, param_values = get_task_parameters_defn_and_params(
                    db_dir=work_dir, row_ids=row_ids
                )
                task_parameters: Any = construct_task_parameters(
                    schema=definition, values=param_values
                )
                task_parameters.lute_config.work_dir = work_dir
                executor._analysis_desc.task_parameters = task_parameters
                executor._row_ids = row_ids
            else:
                logger.critical(
                    "Received start message, but cannot grab TaskParameters!\n"
                    f"Message contents: {msg.contents}"
                )
                return None
            assert executor._analysis_desc.task_parameters is not None
            # Run "before" tasklets
            if executor._tasklets["before"] is not None:
                executor._run_tasklets(when="before")
            # Need to continue since Task._signal_start raises SIGSTOP
            status: int
            _, status = os.waitpid(proc.pid, os.WUNTRACED)
            if os.WIFSTOPPED(status):
                executor._continue(proc)
            executor._task_timeout = (
                executor._analysis_desc.task_parameters.lute_config.task_timeout
            )
            if hasattr(executor._analysis_desc.task_parameters.Config, "set_result"):
                # Tasks may mark a parameter as the result
                # If so, setup the result now.
                executor._set_result_from_parameters()
            logger.info(
                f"Executor: {executor._analysis_desc.task_result.task_name} started"
            )
            executor._analysis_desc.task_result.task_status = TaskStatus.RUNNING
            elog_data: Dict[str, str] = {
                f"{executor._analysis_desc.task_result.task_name} status": "RUNNING",
            }
            post_elog_run_status(elog_data)
            # Tell `maestro` we're RUNNING as well
            if executor._lute_manager_url is not None:
                json_data: Dict[str, Any] = {
                    "managed_task": executor.managed_task_name,
                    "task": executor.task_name,
                    "status": "RUNNING",
                }
                executor._report_to_manager(
                    end_point="status",
                    json_data=json_data,
                )
            return None

        self.add_hook("task_started", task_started)

        def task_failed(
            executor: Executor_T,
            msg: Message,
            proc: Optional[subprocess.Popen] = None,
        ) -> Optional[bool]:
            elog_data: Dict[str, str] = {
                f"{executor._analysis_desc.task_result.task_name} status": "FAILED",
            }
            post_elog_run_status(elog_data)
            return None

        self.add_hook("task_failed", task_failed)

        def task_stopped(
            executor: Executor_T,
            msg: Message,
            proc: Optional[subprocess.Popen] = None,
        ) -> Optional[bool]:
            elog_data: Dict[str, str] = {
                f"{executor._analysis_desc.task_result.task_name} status": "STOPPED",
            }
            post_elog_run_status(elog_data)
            return None

        self.add_hook("task_stopped", task_stopped)

        def task_done(
            executor: Executor_T,
            msg: Message,
            proc: Optional[subprocess.Popen] = None,
        ) -> Optional[bool]:
            elog_data: Dict[str, str] = {
                f"{executor._analysis_desc.task_result.task_name} status": "COMPLETED",
            }
            post_elog_run_status(elog_data)
            return None

        self.add_hook("task_done", task_done)

        def task_cancelled(
            executor: Executor_T,
            msg: Message,
            proc: Optional[subprocess.Popen] = None,
        ) -> Optional[bool]:
            elog_data: Dict[str, str] = {
                f"{executor._analysis_desc.task_result.task_name} status": "CANCELLED",
            }
            post_elog_run_status(elog_data)
            return None

        self.add_hook("task_cancelled", task_cancelled)

        def task_result(
            executor: Executor_T,
            msg: Message,
            proc: Optional[subprocess.Popen] = None,
        ) -> Optional[bool]:
            if isinstance(msg.contents, TaskResult):
                executor._analysis_desc.task_result = msg.contents
                # flake8: noqa: E731
                should_be_printed: Callable[[Any], bool] = lambda x: (
                    isinstance(x, dict)
                    or isinstance(x, str)
                    or (not isinstance(x, ElogSummaryPlots))
                )
                if should_be_printed(executor._analysis_desc.task_result.summary):
                    logger.info(executor._analysis_desc.task_result.summary)
                elif isinstance(executor._analysis_desc.task_result.summary, list):
                    for item in executor._analysis_desc.task_result.summary:
                        if should_be_printed(item):
                            logger.info(item)
                            if self._lute_manager_url is not None:
                                message: str = repr(item)
                                json_data: Dict[str, str] = {
                                    "managed_task": self._m_task_name,
                                    "message": message,
                                }
                                self._report_to_manager(
                                    end_point="log", json_data=json_data
                                )

                logger.info(executor._analysis_desc.task_result.task_status)
            elog_data: Dict[str, str] = {
                f"{executor._analysis_desc.task_result.task_name} status": "COMPLETED",
            }
            post_elog_run_status(elog_data)

            return True

        self.add_hook("task_result", task_result)

        def task_log(
            executor: Executor_T,
            msg: Message,
            proc: Optional[subprocess.Popen] = None,
        ) -> Optional[bool]:
            if isinstance(msg.contents, str):
                # This should be log formatted already
                print(msg.contents, flush=True)
                return True
            return False

        self.add_hook("task_log", task_log)

        def task_request(
            executor: Executor_T,
            msg: Message,
            proc: Optional[subprocess.Popen] = None,
        ) -> Optional[bool]:
            if isinstance(msg, TaskRequestMessage):
                req: TaskRequest = msg.contents
                if req.for_manager:
                    # Task wants to ask something of the workflow manager directly
                    if req.request == "RUNNING_TASKS":
                        if executor._lute_manager_url is not None:
                            # Ask `maestro` for the running Tasks
                            # Response is returned, but ignoring for now
                            resp: Any = executor._report_to_manager(
                                end_point="tasks",
                                json_data=None,
                                method="GET",
                            )
                            for communicator in executor._communicators:
                                if isinstance(communicator, PipeCommunicator):
                                    # Ensure pickle is used.
                                    old_use_pickle: bool = communicator.use_pickle
                                    communicator.use_pickle = True
                                    communicator.write(
                                        Message(contents=resp), proc=proc
                                    )
                                    communicator.use_pickle = old_use_pickle
                            # Immediately read again in this case, since the Task
                            # may have an answer instantly
                            executor._task_loop(proc=proc)  # type: ignore
                else:
                    # Task wants to ask something of another Task
                    # This still goes via the workflow manager. But different APIs
                    ...
            else:
                logger.error(
                    "Task Request improperly formatted. Received message of "
                    f"type: {type(msg)}"
                )
            if __debug__:
                # This means the request gets printed.
                # We don't actually want that in general...
                return False
            return True

        self.add_hook("task_request", task_request)

        def task_metadata(
            executor: Executor_T,
            msg: Message,
            proc: Optional[subprocess.Popen] = None,
        ) -> Optional[bool]:
            if isinstance(msg, TaskMetadataMessage):
                # Maestro just updates all metadata if its provided on any status
                # update. So this can be a simple call.
                if executor._lute_manager_url is not None:
                    json_data: Dict[str, Any] = {
                        "managed_task": executor.managed_task_name,
                        "task": executor.task_name,
                        "status": "RUNNING",
                    }
                    # Add in the Task's metadata
                    json_data.update(msg.contents)
                    executor._report_to_manager(
                        end_point="status",
                        json_data=json_data,
                    )
            else:
                logger.debug("Got metadata signal without metadata message.")

            return None

        self.add_hook("task_metadata", task_metadata)

    def _task_loop(self, proc: subprocess.Popen) -> None:
        """Actions to perform while the Task is running.

        This function is run in the body of a loop until the Task signals
        that its finished.
        """
        # Some hooks may ask that the rest of the task loop be skipped (continued)
        should_continue: Optional[bool]
        for communicator in self._communicators:
            while True:
                # We'll try to track any sort of signal of Task life.
                # Any message at ALL will be used as a proxy of life (logs, signals, etc)
                msg: Message = communicator.read(proc)
                if msg.signal is not None and msg.signal.upper() in LUTE_SIGNALS:
                    # Have signal? ---> Consider it a sign of life
                    self._task_last_update = time.monotonic()
                    hook: Hook = getattr(self.Hooks, msg.signal.lower())
                    should_continue = hook(self, msg, proc)
                    if should_continue:
                        continue

                if msg.contents is not None:
                    # Have message contents? ---> Consider it a sign of life
                    self._task_last_update = time.monotonic()
                    if isinstance(msg.contents, str) and msg.contents != "":
                        logger.info(msg.contents)
                        if self._lute_manager_url is not None:
                            message: str = msg.contents
                            json_data: Dict[str, str] = {
                                "managed_task": self._m_task_name,
                                "message": message,
                            }
                            self._report_to_manager(
                                end_point="log", json_data=json_data
                            )
                    elif isinstance(msg.contents, TaskParametersDBReference):
                        # We will log the actual reconstructed TaskParameters object
                        # instead of the raw message. The raw message only contains
                        # the pointers for reconstructing the object.
                        logger.info(self._analysis_desc.task_parameters)
                        if self._lute_manager_url is not None:
                            message = repr(msg.contents)
                            json_data = {
                                "managed_task": self._m_task_name,
                                "message": message,
                            }
                            self._report_to_manager(
                                end_point="log", json_data=json_data
                            )
                    elif not isinstance(msg.contents, str):
                        logger.info(msg.contents)
                        if self._lute_manager_url is not None:
                            message = repr(msg.contents)
                            json_data = {
                                "managed_task": self._m_task_name,
                                "message": message,
                            }
                            self._report_to_manager(
                                end_point="log", json_data=json_data
                            )
                if not communicator.has_messages:
                    break

    def _finalize_task(self, proc: subprocess.Popen) -> None:
        """Any actions to be performed after the Task has ended.

        Examples include a final clearing of the pipes, retrieving results,
        reporting to third party services, etc.
        """
        self._task_loop(proc)  # Perform a final read.

    def _process_results(self) -> None:
        """Performs result processing.

        Actions include:
        - For `ElogSummaryPlots`, will save the summary plot to the appropriate
            directory for display in the eLog.
        """
        task_result: TaskResult = self._analysis_desc.task_result
        self._process_result_payload(task_result.payload)
        self._process_result_summary(task_result.summary)

    def _process_result_payload(self, payload: Any) -> None:
        if self._analysis_desc.task_parameters is None:
            logger.error(
                "Please run Task before using this method! (_process_result_payload). "
                "If you did run a Task, it may have failed immediately!"
            )
            return
        new_payload: Optional[str]
        if isinstance(payload, ElogSummaryPlots):
            new_payload = self._process_elog_plot(payload)
            if new_payload is not None:
                self._analysis_desc.task_result.payload = new_payload
        elif isinstance(payload, list) or isinstance(payload, tuple):
            new_payload = ""
            for item in payload:
                if isinstance(item, ElogSummaryPlots):
                    ret: Optional[str] = self._process_elog_plot(item)
                    if ret is not None:
                        new_payload = ";".join(filter(None, (new_payload, ret)))
            if new_payload != "":
                self._analysis_desc.task_result.payload = new_payload
        elif isinstance(payload, str):
            ...
            # May be a path to a file...
            # schemas: Optional[str] = self._analysis_desc.task_result.impl_schemas
            # Should also check `impl_schemas` to determine what to do with path

    def _process_elog_plot(self, plots: ElogSummaryPlots) -> Optional[str]:
        """Process an ElogSummaryPlots

        Writes out the eLog summary plot for display and returns the path of
        where the plots were written out so they can be stored as the result
        payload.

        ElogSummaryPlots objects already convert the plots to a byte stream
        which can be directly written to an HTML file.

        Args:
            plots (ElogSummaryPlots): The plots dataclass.

        Returns:
            path (str): Path the plots were written out to.
        """
        if self._analysis_desc.task_parameters is None:
            logger.error(
                "Please run Task before using this method! (_process_elog_plot). "
                "If you did run a Task, it may have failed immediately!"
            )
            return None
        # ElogSummaryPlots has figures and a display name
        # display name also serves as a path.
        expmt: str = self._analysis_desc.task_parameters.lute_config.experiment
        base_path: str = f"/sdf/data/lcls/ds/{expmt[:3]}/{expmt}/stats/summary"
        full_path: str = f"{base_path}/{plots.display_name}"
        if not os.path.isdir(full_path):
            os.makedirs(full_path)

        path: str = f"{full_path}/report.html"
        with open(f"{full_path}/report.html", "wb") as f:
            f.write(plots.figures)

        return path

    def _process_result_summary(self, summary: Any) -> None:
        """Process an object destined for the results summary.

        Args:
            summary (Any): The object to be set as a summary. If a dictionary
                it is assumed to be a set of key/value pairs to be written out
                as run parameters in the eLog. If a list each item is processed
                individually.
        """
        if self._analysis_desc.task_parameters is None:
            logger.error(
                "Please run Task before using this method! (_process_result_summary). "
                "If you did run a Task, it may have failed immediately!"
            )
            return
        if isinstance(summary, dict):
            # Assume dict is key: value pairs of eLog run parameters to post
            self._analysis_desc.task_result.summary = self._process_summary_run_params(
                summary
            )
        elif isinstance(summary, list) or isinstance(summary, tuple):
            new_summary_str: str = ""
            for item in summary:
                if isinstance(item, dict):
                    ret: str = self._process_summary_run_params(item)
                    new_summary_str = ";".join(filter(None, (new_summary_str, ret)))
                elif isinstance(item, ElogSummaryPlots):
                    plot_path: Optional[str] = self._process_elog_plot(item)
                    new_summary_str = ";".join(
                        filter(None, (new_summary_str, plot_path))
                    )
            self._analysis_desc.task_result.summary = new_summary_str
        elif isinstance(summary, str):
            ...
        else:
            ...

    def _process_summary_run_params(self, params: Dict[str, str]) -> str:
        """Process a dictionary of run parameters to be posted to the eLog.

        Args:
            params (Dict[str, str]): Key/value pairs to be posted as run parameters.

        Returns:
            summary_str (str): New string of key/value pairs to be stored in
                summary field of the database.
        """
        if self._analysis_desc.task_parameters is None:
            logger.error(
                "Please run Task before using this method! (_process_summary_run_params). "
                "If you did run a Task, it may have failed immediately!"
            )
            return ""
        exp: str = self._analysis_desc.task_parameters.lute_config.experiment
        run: int = int(self._analysis_desc.task_parameters.lute_config.run)
        logger.debug("Posting eLog run parameters.")
        try:
            post_elog_run_table(exp, run, params)
        except Exception as err:
            logger.error(f"Unable to post run parameters! Error: {err}")
        post_elog_run_status(params)
        summary_str: str = ";".join(f"{key}: {value}" for key, value in params.items())
        return summary_str


class MPIExecutor(Executor):
    """Runs first-party Tasks that require MPI.

    This Executor is otherwise identical to the standard Executor, except it
    uses `mpirun` for `Task` submission. Currently this Executor assumes a job
    has been submitted using SLURM as a first step. It will determine the number
    of MPI ranks based on the resources requested. As a fallback, it will try
    to determine the number of local cores available for cases where a job has
    not been submitted via SLURM. On S3DF, the second determination mechanism
    should accurately match the environment variable provided by SLURM indicating
    resources allocated.

    This Executor will submit the Task to run with a number of processes equal
    to the total number of cores available minus 1. A single core is reserved
    for the Executor itself. Note that currently this means that you must submit
    on 3 cores or more, since MPI requires a minimum of 2 ranks, and the number
    of ranks is determined from the cores dedicated to Task execution.

    Methods:
        _submit_cmd: Run the task as a subprocess using `mpirun`.
    """

    def _submit_cmd(self, executable_path: str, params: str) -> str:
        """Override submission command to use `mpirun`

        Args:
            executable_path (str): Path to the LUTE subprocess script.

            params (str): String of formatted command-line arguments.

        Returns:
            cmd (str): Appropriately formatted command for this Executor.
        """
        py_cmd: str = ""
        nprocs: int = max(
            int(os.environ.get("SLURM_NPROCS", len(os.sched_getaffinity(0)))) - 1, 1
        )
        mpi_cmd: str = f"mpirun -np {nprocs} --map-by core"
        print(self._py_executable, flush=True)
        if __debug__:
            py_cmd = (
                f"{self._py_executable} -B -u -m mpi4py.run {executable_path} {params}"
            )
        else:
            py_cmd = (
                f"{self._py_executable} -OB -u -m mpi4py.run {executable_path} {params}"
            )

        cmd: str = f"{mpi_cmd} {py_cmd}"
        return cmd
