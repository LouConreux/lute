"""Base classes and functions for handling `Task` execution.

Executors run a `Task` as a subprocess and handle all communication with other
services, e.g., the eLog. They accept specific handlers to override default
stream parsing.

Event handlers/hooks are implemented as standalone functions which can be added
to an Executor.


Classes:
    AnalysisConfig: Data class for holding a managed Task's configuration.

    BaseExecutor: Abstract base class from which all Executors are derived.

    Executor: Default Executor implementing all basic functionality and IPC.

    BinaryExecutor: Can execute any arbitrary binary/command as a managed task
        within the framework provided by LUTE.

Exceptions
----------

"""

__all__ = ["BaseExecutor", "Executor", "MPIExecutor"]
__author__ = "Gabriel Dorlhiac"

import _io
import logging
import subprocess
import time
import os
import signal
from typing import Dict, Callable, List, Optional
from typing_extensions import Self
from abc import ABC, abstractmethod
import warnings
import copy

from .ipc import *
from ..tasks.task import *
from ..tasks.dataclasses import *
from ..io.models.base import TaskParameters
from ..io.db import record_analysis_db

if __debug__:
    warnings.simplefilter("default")
    os.environ["PYTHONWARNINGS"] = "default"
    logging.basicConfig(level=logging.DEBUG)
    logging.captureWarnings(True)
else:
    logging.basicConfig(level=logging.INFO)
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

logger: logging.Logger = logging.getLogger(__name__)


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

        execute_task(): Run the task as a subprocess.
    """

    class Hooks:
        """A container class for the Executor's event hooks.

        There is a corresponding function (hook) for each event/signal. Each
        function takes two parameters - a reference to the Executor (self) and
        a reference to the Message (msg) which includes the corresponding
        signal.
        """

        def no_pickle_mode(self: Self, msg: Message):
            ...

        def task_started(self: Self, msg: Message):
            ...

        def task_failed(self: Self, msg: Message):
            ...

        def task_stopped(self: Self, msg: Message):
            ...

        def task_done(self: Self, msg: Message):
            ...

        def task_cancelled(self: Self, msg: Message):
            ...

        def task_result(self: Self, msg: Message):
            ...

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
        task_parameters: TaskParameters = TaskParameters()
        task_env: Dict[str, str] = os.environ.copy()
        self._communicators: List[Communicator] = communicators
        communicator_desc: List[str] = []
        for comm in self._communicators:
            comm.stage_communicator()
            communicator_desc.append(str(comm))

        self._analysis_desc: DescribedAnalysis = DescribedAnalysis(
            task_result=result,
            task_parameters=task_parameters,
            task_env=task_env,
            poll_interval=poll_interval,
            communicator_desc=communicator_desc,
        )

    def add_hook(self, event: str, hook: Callable[[Self, Message], None]) -> None:
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

    def update_environment(
        self, env: Dict[str, str], update_path: str = "prepend"
    ) -> None:
        """Update the stored set of environment variables.

        These are passed to the subprocess to setup its environment.

        Args:
            env (Dict[str, str]): A dictionary of "VAR":"VALUE" pairs of
                environment variables to be added to the subprocess environment.
                If any variables already exist, the new variables will
                overwrite them (except PATH, see below).

            update_path (str): If PATH is present in the new set of variables,
                this argument determines how the old PATH is dealt with. There
                are three options:
                * "prepend" : The new PATH values are prepended to the old ones.
                * "append" : The new PATH values are appended to the old ones.
                * "overwrite" : The old PATH is overwritten by the new one.
                "prepend" is the default option. If PATH is not present in the
                current environment, the new PATH is used without modification.
        """
        if "PATH" in env:
            sep: str = os.pathsep
            if update_path == "prepend":
                env[
                    "PATH"
                ] = f"{env['PATH']}{sep}{self._analysis_desc.task_env['PATH']}"
            elif update_path == "append":
                env[
                    "PATH"
                ] = f"{self._analysis_desc.task_env['PATH']}{sep}{env['PATH']}"
            elif update_path == "overwrite":
                pass
            else:
                raise ValueError(
                    (
                        f"{update_path} is not a valid option for `update_path`!"
                        " Options are: prepend, append, overwrite."
                    )
                )
        self._analysis_desc.task_env.update(env)

    def source_env(self, env: str) -> None:
        """Source a script.

        Unlike `update_environment` this method sources a new file.

        Args:
            env (str): Path to the script to source.
        """
        import sys

        if not os.path.exists(env):
            logger.info(f"Cannot source environment from {env}!")
            return

        script: str = (
            f"set -a\n"
            f'source "{env}" >/dev/null\n'
            f'{sys.executable} -c "import os; print(dict(os.environ))"\n'
        )
        logger.info(f"Sourcing file {env}")
        o, e = subprocess.Popen(
            ["bash", "-c", script], stdout=subprocess.PIPE
        ).communicate()
        new_environment: Dict[str, str] = eval(o)
        self._analysis_desc.task_env = new_environment

    def _pre_task(self) -> None:
        """Any actions to be performed before task submission.

        This method may or may not be used by subclasses. It may be useful
        for logging etc.
        """
        ...

    def _submit_task(self, cmd: str) -> subprocess.Popen:
        proc: subprocess.Popen = subprocess.Popen(
            cmd.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self._analysis_desc.task_env,
        )
        os.set_blocking(proc.stdout.fileno(), False)
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

    def execute_task(self) -> None:
        """Run the requested Task as a subprocess."""
        lute_path: Optional[str] = os.getenv("LUTE_PATH")
        if lute_path is None:
            logger.debug("Absolute path to subprocess.py not found.")
            lute_path = os.path.abspath(f"{os.path.dirname(__file__)}/../..")
            os.environ["LUTE_PATH"] = lute_path
        executable_path: str = f"{lute_path}/subprocess_task.py"
        config_path: str = self._analysis_desc.task_env["LUTE_CONFIGPATH"]
        params: str = f"-c {config_path} -t {self._analysis_desc.task_result.task_name}"

        cmd: str = ""
        if __debug__:
            cmd = f"python -B {executable_path} {params}"
        else:
            cmd = f"python -OB {executable_path} {params}"

        proc: subprocess.Popen = self._submit_task(cmd)

        while self._task_is_running(proc):
            self._task_loop(proc)
            time.sleep(self._analysis_desc.poll_interval)

        os.set_blocking(proc.stdout.fileno(), True)
        os.set_blocking(proc.stderr.fileno(), True)

        self._finalize_task(proc)
        proc.stdout.close()
        proc.stderr.close()
        proc.wait()
        if ret := proc.returncode:
            logger.info(f"Task failed with return code: {ret}")
            self._analysis_desc.task_result.task_status = TaskStatus.FAILED
        elif self._analysis_desc.task_result.task_status == TaskStatus.RUNNING:
            # Ret code is 0, no exception was thrown, task forgot to set status
            self._analysis_desc.task_result.task_status = TaskStatus.COMPLETED
            logger.debug(f"Task did not change from RUNNING status. Assume COMPLETED.")
        self._store_configuration()
        for comm in self._communicators:
            comm.clear_communicator()

    def _store_configuration(self) -> None:
        """Store configuration and results in the LUTE database."""
        record_analysis_db(copy.deepcopy(self._analysis_desc))

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
        return proc.poll() is None and is_running

    def _stop(self, proc: subprocess.Popen) -> None:
        """Stop the Task subprocess."""
        os.kill(proc.pid, signal.SIGTSTP)
        self._analysis_desc.task_result.task_status = TaskStatus.STOPPED

    def _continue(self, proc: subprocess.Popen) -> None:
        """Resume a stopped Task subprocess."""
        os.kill(proc.pid, signal.SIGCONT)
        self._analysis_desc.task_result.task_status = TaskStatus.RUNNING


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

        def no_pickle_mode(self: Executor, msg: Message):
            for idx, communicator in enumerate(self._communicators):
                if isinstance(communicator, PipeCommunicator):
                    self._communicators[idx] = PipeCommunicator(
                        Party.EXECUTOR, use_pickle=False
                    )

        self.add_hook("no_pickle_mode", no_pickle_mode)

        def task_started(self: Executor, msg: Message):
            if isinstance(msg.contents, TaskParameters):
                self._analysis_desc.task_parameters = msg.contents
            logger.info(
                f"Executor: {self._analysis_desc.task_result.task_name} started"
            )
            self._analysis_desc.task_result.task_status = TaskStatus.RUNNING

        self.add_hook("task_started", task_started)

        def task_failed(self: Executor, msg: Message):
            ...

        self.add_hook("task_failed", task_failed)

        def task_stopped(self: Executor, msg: Message):
            ...

        self.add_hook("task_stopped", task_stopped)

        def task_done(self: Executor, msg: Message):
            ...

        self.add_hook("task_done", task_done)

        def task_cancelled(self: Executor, msg: Message):
            ...

        self.add_hook("task_cancelled", task_cancelled)

        def task_result(self: Executor, msg: Message):
            if isinstance(msg.contents, TaskResult):
                self._analysis_desc.task_result = msg.contents
                logger.info(self._analysis_desc.task_result.summary)
                logger.info(self._analysis_desc.task_result.task_status)

        self.add_hook("task_result", task_result)

    def _task_loop(self, proc: subprocess.Popen) -> None:
        """Actions to perform while the Task is running.

        This function is run in the body of a loop until the Task signals
        that its finished.
        """
        for communicator in self._communicators:
            msg: Message = communicator.read(proc)
            if msg.signal is not None and msg.signal.upper() in LUTE_SIGNALS:
                hook: Callable[[None], None] = getattr(self.Hooks, msg.signal.lower())
                hook(self, msg)
            if msg.contents is not None:
                if isinstance(msg.contents, str) and msg.contents != "":
                    logger.info(msg.contents)
                elif not isinstance(msg.contents, str):
                    logger.info(msg.contents)

    def _finalize_task(self, proc: subprocess.Popen) -> None:
        """Any actions to be performed after the Task has ended.

        Examples include a final clearing of the pipes, retrieving results,
        reporting to third party services, etc.
        """
        self._task_loop(proc)  # Perform a final read.


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
    for the Executor itself.

    Methods:
        execute_task(): Run the task as a subprocess using `mpirun`.
    """

    def execute_task(self) -> None:
        """Run the requested Task as a subprocess."""
        lute_path: Optional[str] = os.getenv("LUTE_PATH")
        if lute_path is None:
            logger.debug("Absolute path to subprocess.py not found.")
            lute_path = os.path.abspath(f"{os.path.dirname(__file__)}/../..")
            os.environ["LUTE_PATH"] = lute_path
        executable_path: str = f"{lute_path}/subprocess_task.py"
        config_path: str = self._analysis_desc.task_env["LUTE_CONFIGPATH"]
        params: str = f"-c {config_path} -t {self._analysis_desc.task_result.task_name}"

        py_cmd: str = ""
        mpi_cmd: str = f"mpirun -np {int(os.environ.get('SLURM_NPROCS', len(os.sched_getaffinity(0)))) - 1}"
        if __debug__:
            py_cmd = f"python -B -u -m mpi4py.run {executable_path} {params}"
        else:
            py_cmd = f"python -OB -u -m mpi4py.run {executable_path} {params}"

        cmd: str = f"{mpi_cmd} {py_cmd}"
        proc: subprocess.Popen = self._submit_task(cmd)

        while self._task_is_running(proc):
            self._task_loop(proc)
            time.sleep(self._analysis_desc.poll_interval)

        os.set_blocking(proc.stdout.fileno(), True)
        os.set_blocking(proc.stderr.fileno(), True)

        self._finalize_task(proc)
        proc.stdout.close()
        proc.stderr.close()
        proc.wait()
        if ret := proc.returncode:
            logger.info(f"Task failed with return code: {ret}")
            self._analysis_desc.task_result.task_status = TaskStatus.FAILED
        elif self._analysis_desc.task_result.task_status == TaskStatus.RUNNING:
            # Ret code is 0, no exception was thrown, task forgot to set status
            self._analysis_desc.task_result.task_status = TaskStatus.COMPLETED
            logger.debug(f"Task did not change from RUNNING status. Assume COMPLETED.")
        self._store_configuration()
        for comm in self._communicators:
            comm.clear_communicator()
