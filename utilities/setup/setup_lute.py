"""Script to setup LUTE and workflow definitions."""

__author__ = "Gabriel Dorlhiac"

import argparse
import logging
import os
import requests
import shutil
import subprocess
import sys
from typing import List, Dict, Any, Optional

from krtc import KerberosTicket  # type: ignore

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "MEC": {
        "SmallDataProducer": {
            "nodes": 1,
            "ntasks_per_node": 10,
        },
        "SmallDataProducer2": {
            "nodes": 1,
            "ntasks_per_node": 10,
        },
        # 63 ranks as elsewhere; 119 (edit slurm_params in the DAG) gives more seeds per quad.
        "BayFAIOptimizer": {
            "nodes": 1,
            "ntasks_per_node": 64,
        },
        "BayFAIOptimizer2": {
            "nodes": 1,
            "ntasks_per_node": 64,
        },
    },
    "ELSE": {
        "SmallDataProducer": {
            "nodes": 2,
            "ntasks_per_node": 50,
        },
        "SmallDataProducer2": {
            "nodes": 2,
            "ntasks_per_node": 50,
        },
        # 63 ranks: 119 did not help MFX and doubles memory.
        "BayFAIOptimizer": {
            "nodes": 1,
            "ntasks_per_node": 64,
        },
        "BayFAIOptimizer2": {
            "nodes": 1,
            "ntasks_per_node": 64,
        },
    },
}


# Available Python interpreters for multiversioning.
PYTHON_INTERPRETERS: Dict[str, str] = {
    "3.9": "/sdf/group/lcls/ds/ana/sw/conda2/inst/bin/python3.9",
    "3.11": "/sdf/group/lcls/ds/ana/sw/conda2-v3/inst/bin/python3.11",
}


def _run_subprocess_log(
    cmd: List[str], env: Optional[Dict[str, str]] = None, cwd: Optional[str] = None
) -> None:
    """Run a subprocess with logging.

    Args:
        cmd (List[str]): Command to run as a list of strings.

        env (Optional[Dict[str, str]]): Environment to run the command in.

        cwd (Optional[str]):  Working directory to run the command in.
    """
    global logger

    out: str
    err: str
    out, err = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        env=env,
        cwd=cwd,
    ).communicate()
    if out:
        logger.info(out)
    if err:
        logger.info(err)


def touch(full_path: str) -> None:
    """Touch a file.

    Args:
        full_path (str): The full path to the file to touch.
    """
    cmd: List[str] = ["touch", full_path]
    _run_subprocess_log(cmd)


def database_setup(full_path: str) -> None:
    """Touch a file.

    Args:
        full_path (str): The full path to the file to touch.
    """
    if os.path.exists(full_path):
        logger.error(
            f"LUTE database already exists at: {full_path}. Will not overwrite, exiting."
        )
        sys.exit(-1)
    touch(full_path)
    os.chmod(full_path, 0o664)


def git_clone(repo: str, location: str, tag: str) -> None:
    """Clone a git repository.

    Will not overwrite a directory of there is already a folder at the specified
    location.
    Args:
        repo (str): Name of the repository to clone. Should be specified as:
            "<user_or_organization>/<repository_name>"

        location (str): Path to the location to clone to.
    """
    global logger

    repo_only: str = repo.split("/")[1]
    if os.path.exists(f"{location}/{repo_only}"):
        logger.debug(
            f"Repository {repo} already exists at {location}. Will not overwrite."
        )
        return
    cmd: List[str] = [
        "git",
        "clone",
        f"https://github.com/{repo}.git",
        location,
    ]
    _run_subprocess_log(cmd)

    cmd = ["git", "checkout", tag]
    _run_subprocess_log(cmd, cwd=location)


def run_build_script(lute_path: str) -> None:
    """Run the LUTE build script.

    Args:
        lute_path (str): The path to the LUTE installation to build.
    """

    cmd: List[str] = ["./build.sh", "-e", "-r"]
    logger.info(f"Building LUTE at {lute_path}. This may take a few minutes...")
    _run_subprocess_log(cmd, cwd=lute_path)


class LuteEnvBuilder:
    """Manages creation of LUTE virtual environments.

    Encapsulates the standard directory naming scheme, package installation,
    and environment variable setup for LUTE venvs. Supports multiversioning
    by creating venvs for different Python interpreters.

    Args:
        lute_version (str): The version of LUTE to install in the venvs.

        env_dir (str): The base directory to create venvs in. Each venv will be
            created in a subdirectory named according to the Python version.
    """

    def __init__(self, lute_version: str, env_dir: str) -> None:
        self.lute_version = lute_version
        self.env_dir = os.path.abspath(env_dir)
        self._created: Dict[str, str] = {}

    def venv_path(self, python_version: str) -> str:
        """Standard venv directory path for a given Python version."""
        version = python_version.replace(".", "")
        return os.path.join(self.env_dir, f"lute_env_py{version}")

    def bin_path(self, python_version: str) -> str:
        """Path to the venv's bin directory."""
        return os.path.join(self.venv_path(python_version), "bin")

    def site_packages_path(self, python_version: str) -> str:
        """Path to the venv's site-packages directory."""
        return os.path.join(
            self.venv_path(python_version),
            "lib",
            f"python{python_version}",
            "site-packages",
        )

    def create(self, python_version: str) -> str:
        """Create a LUTE venv for the given Python version.

        Looks up the interpreter from PYTHON_INTERPRETERS, creates the venv
        using the standard naming scheme, installs lute-lcls, and writes
        an activation script that sets LUTE_ environment variables.

        Args:
            python_version: Version string (e.g. "3.9", "3.11").

        Returns:
            The absolute path to the created venv directory.
        """
        python_path = PYTHON_INTERPRETERS.get(python_version)
        if python_path is None:
            logger.warning(
                f"No interpreter configured for Python {python_version}. "
                f"Available: {list(PYTHON_INTERPRETERS.keys())}"
            )
            sys.exit(-1)

        env_dir = self.venv_path(python_version)
        venv_python = os.path.join(env_dir, "bin", "python")

        if os.path.exists(env_dir):
            logger.info(f"Virtual environment already exists at {env_dir}. Reusing.")
        else:
            logger.info(f"Creating python{python_version} env at {env_dir}...")
            cmd: List[str] = [python_path, "-m", "venv", env_dir]
            _run_subprocess_log(cmd)

        # Install lute-lcls.
        extra_index_args: List[str] = []
        if self.lute_version == "dev":
            pkg_spec = "lute-lcls"
            extra_index_args = [
                "--extra-index-url",
                "https://slac-lcls.github.io/lute/wheels",
                "--pre",
            ]
        else:
            pkg_spec = f"lute-lcls=={self.lute_version}"
        logger.info(f"Installing {pkg_spec} into {env_dir}...")
        _run_subprocess_log(
            [venv_python, "-m", "pip", "install", "--upgrade", "pip"],
        )
        _run_subprocess_log(
            [venv_python, "-m", "pip", "install", "--ignore-installed", pkg_spec]
            + extra_index_args,
        )

        # Write LUTE environment variables into the activation script
        self._write_env_vars(env_dir, python_version)
        modify_permissions(env_dir)

        self._created[python_version] = env_dir
        return env_dir

    def _write_env_vars(self, env_dir: str, python_version: str) -> None:
        """Append LUTE_ environment variables to the venv activate script."""
        activate_path = os.path.join(env_dir, "bin", "activate")
        env_vars = (
            f"\n# LUTE environment variables\n"
            f'export LUTE_VIRTUAL_ENV_PY{python_version.replace(".", "")}="{env_dir}/bin/python"\n'
            f'export LUTE_NEW_PYVER="python{python_version}"\n'
        )
        with open(activate_path, "a") as f:
            f.write(env_vars)


def pip_install(src_dir: str, install_dir: Optional[str] = None) -> None:
    """Install from a source directory to an optionally specified directory.

    If no prefix (`install_dir`) is provided, this will install to the current
    packages directory determined by, e.g., conda environment, etc.

    If a prefix is provided this command will also create the directory (or
    multiple directories) if necessary.

    Args:
        src_dir (str): Directory with the source code and setup.py.

        install_dir (Optional[str]): Optionally provide a directory to install
            install to.
    """
    cmd: List[str] = [
        "pip",
        "install",
        "--no-deps",
        src_dir,
        f'--prefix="{install_dir}"',
    ]
    logger.info(f"Attempting to install from: {src_dir} to: {install_dir}")
    env: Dict[str, str] = os.environ.copy()
    env["PATH"] = (
        f"/sdf/group/lcls/ds/ana/sw/conda1/inst/envs/ana-4.0.63-py3/bin:{env['PATH']}"
    )
    if install_dir is not None:
        cmd.append(f"--prefix={install_dir}")
        if not os.path.exists(install_dir):
            mkdir_cmd: List[str] = ["mkdir", "-p", install_dir]
            _run_subprocess_log(mkdir_cmd)
    _run_subprocess_log(cmd, env=env)


def inplace_sed(in_file: str, pattern: str) -> None:
    """Perform an in-place operation on a file using sed.

    Args:
        in_file (str): Path to the file to perform the substitution on.

        pattern (str): Operation. E.g. substitute with "s/old_text/new_text/g"
    """
    cmd: List[str] = ["sed", "-i", pattern, in_file]
    _run_subprocess_log(cmd)


def modify_permissions(lute_path: str) -> None:
    """Recursively set permissions for a LUTE installation.

    Args:
        lute_path (str): Root path to apply permissions to.
    """
    os.chmod(lute_path, 0o775)
    for root, dirs, files in os.walk(lute_path):
        for d in dirs:
            dir_path = os.path.join(root, d)
            if not os.path.islink(dir_path):
                os.chmod(dir_path, 0o775)
        for f in files:
            file_path = os.path.join(root, f)
            if not os.path.islink(file_path):
                os.chmod(file_path, 0o775)


def update_dag_params(
    dag_path: str,
    partition: str,
    account: str,
    extra_slurm_params: str,
    default_config: dict,
    user_slurm_params: str,
) -> None:
    """Update slurm_params in a DAG file in place.

    For tasks listed in DEFAULT_CONFIG, use the user's flags and fill unset nodes/ntasks from it.
    For all other tasks, forward the user-provided SLURM parameters verbatim.

    Args:
        dag_path (str): Path to the DAG file.

        partition (str): SLURM partition.

        account (str): SLURM account.

        extra_slurm_params (str): Additional SLURM parameters, forwarded as-is.

        user_slurm_params (str): The flags the user passed, without the prompted defaults.
    """
    with open(dag_path, "r") as f:
        lines: List[str] = f.readlines()

    result: List[str] = []
    current_task: Optional[str] = None

    for line in lines:
        stripped = line.lstrip()
        task_name = None
        if stripped.startswith("task_name:"):
            task_name = stripped
        elif stripped.startswith("- task_name:"):
            task_name = stripped[2:]  # Strip the "- " prefix
        if task_name is not None:
            # Extract task name (handles both quoted and unquoted)
            task_value = task_name.split(":", 1)[1].strip().strip("\"'")
            current_task = task_value

        if stripped.startswith("slurm_params:"):
            indent = line[: len(line) - len(stripped)]
            if current_task and current_task in default_config:
                cfg = default_config[current_task]
                # The task's defaults only fill in what the user did not set.
                flags = [f"--account={account}", f"--partition={partition}"]
                flags += [user_slurm_params] if user_slurm_params else []
                if "--ntasks" not in user_slurm_params:
                    flags.append(f"--ntasks-per-node={cfg['ntasks_per_node']}")
                if "--nodes" not in user_slurm_params:
                    flags.append(f"--nodes={cfg['nodes']}")
                params = " ".join(flags)
            else:
                params = (
                    f"--account={account} --partition={partition} {extra_slurm_params}"
                )
            result.append(f"{indent}slurm_params: '{params}'\n")
        else:
            result.append(line)

    with open(dag_path, "w") as f:
        f.writelines(result)

    logger.info(f"Updated slurm_params in DAG file: {dag_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="setup_lute",
        description="Setup LUTE work space and eLog workflows for an experiment.",
        epilog="Refer to https://github.com/slac-lcls/lute for more information.",
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Turn on verbose logging."
    )
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        help="Experiment to perform setup for.",
        required=True,
    )
    parser.add_argument(
        "-fb",
        "--fresh_build",
        help=(
            "Install a new install of LUTE in the experiment folder via cloning "
            "fresh repository and building. This allows for local modifications "
            "of code. Otherwise, the central installation will be used which "
            "cannot be modified."
        ),
        action="store_true",
    )
    parser.add_argument(
        "-fi",
        "--fresh_install",
        help=(
            "Install a new version of LUTE in the experiment folder via isolate "
            "virtual environment. This allows for local modifications of code. "
            "Otherwise, the central installation will be used which cannot be "
            "modified."
        ),
        action="store_true",
    )
    parser.add_argument(
        "-D",
        "--directory",
        type=str,
        help=(
            "Subdirectory name within the experiment results folder to use for "
            "LUTE output and LUTE fresh install if specified. If not specified, "
            "will use results folder directly."
        ),
        default="",
    )
    parser.add_argument(
        "--test", help="Use test Airflow instance.", action="store_true"
    )
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        help=(
            "Version of LUTE to use. Corresponds to release tag or `dev`. "
            "Defaults to `dev`. If `fresh_install`, this fetches the latest "
            "version of LUTE and adds the nightly build wheels."
        ),
        default="dev",
    )
    parser.add_argument(
        "-W",
        "--workflow",
        type=str,
        nargs="+",
        action="extend",
        help="Which analysis workflow(s) to run. E.g. -W smd bayfai.",
    )
    args: argparse.Namespace
    extra_args: List[str]  # May have additional SLURM arguments
    args, extra_args = parser.parse_known_args()

    hutch: str = args.experiment[:3]

    workflow_names: List[str] = args.workflow if args.workflow else ["smd"]

    results_dir: str = f"/sdf/data/lcls/ds/{hutch}/{args.experiment}/results"
    if args.directory != "":
        results_dir = f"{results_dir}/{args.directory}"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, mode=0o777)
            os.chmod(results_dir, 0o777)
    lute_path: str
    arp_executable: str
    launch_executable: str
    std_hutch_config: str
    std_test_config: str
    if args.fresh_install:
        version: str = args.version
        # Create a venv for each requested Python version
        env_dir = f"{results_dir}/lute_envs"
        builder = LuteEnvBuilder(lute_version=version, env_dir=env_dir)
        for pv in PYTHON_INTERPRETERS.keys():
            builder.create(pv)
        modify_permissions(env_dir)

        # Use the first requested version as the primary for workflow setup
        primary_version: str = list(PYTHON_INTERPRETERS.keys())[0]
        lute_path = builder.venv_path(primary_version)
        venv_bin: str = builder.bin_path(primary_version)
        arp_executable = f"{venv_bin}/submit_launch_slurm.sh"
        launch_executable = f"{venv_bin}/launch_slurm"
        # In a pip-installed venv, config lives inside site-packages
        site_packages: str = builder.site_packages_path(primary_version)
        std_hutch_config = f"{site_packages}/config/{hutch}.yaml"
        std_test_config = f"{site_packages}/config/test.yaml"
    elif args.fresh_build:
        lute_path = f"{results_dir}/lute"
        git_clone("slac-lcls/lute", lute_path, args.version)
        run_build_script(lute_path)
        modify_permissions(lute_path)
        arp_executable = f"{lute_path}/install/bin/submit_launch_slurm.sh"
        launch_executable = f"{lute_path}/install/bin/launch_slurm"
        std_hutch_config = f"{lute_path}/config/{hutch}.yaml"
        std_test_config = f"{lute_path}/config/test.yaml"
    else:
        lute_path = f"/sdf/group/lcls/ds/tools/lute/{args.version}/lute"
        arp_executable = f"{lute_path}/install/bin/submit_launch_slurm.sh"
        launch_executable = f"{lute_path}/install/bin/launch_slurm"
        std_hutch_config = f"{lute_path}/config/{hutch}.yaml"
        std_test_config = f"{lute_path}/config/test.yaml"

    central_install: str = f"/sdf/group/lcls/ds/tools/lute/{args.version}/lute"
    lute_output_dir: str = f"{results_dir}/lute_output"
    if not os.path.exists(lute_output_dir):
        os.makedirs(lute_output_dir, mode=0o777)
        os.chmod(lute_output_dir, 0o777)
    config_path: str = f"{lute_output_dir}/{hutch}_lute.yaml"
    if os.path.exists(config_path):
        logger.error(
            f"Configuration YAML already exists at: {config_path}. Will not overwrite, exiting."
        )
        sys.exit(-1)
    if not os.path.exists(std_hutch_config):
        shutil.copy(std_test_config, config_path)
    else:
        shutil.copy(std_hutch_config, config_path)
    os.chmod(config_path, 0o666)
    # Substitute the work_dir in LUTE's config to the experiment results folder.
    sed_pattern: str = rf's|work_dir:\(.*\)|work_dir: \\"{lute_output_dir}\\"|g'
    inplace_sed(config_path, sed_pattern)

    database_setup(f"{lute_output_dir}/lute.db")  # Setup permissions on database

    # Check for partition and account. If not provided, prompt the user to use defaults.
    partition: str = "milano"
    account: str = f"lcls:{args.experiment}"
    has_partition: bool = False
    has_account: bool = False
    extra_slurm_args: List[str] = []
    for arg in extra_args:
        if arg.startswith("--partition="):
            partition = arg.split("=", 1)[1]
            has_partition = True
        elif arg.startswith("--account="):
            account = arg.split("=", 1)[1]
            has_account = True
        else:
            extra_slurm_args.append(arg)

    if not has_partition:
        logger.warning(
            f"No queue/partition provided. Defaulting to {partition}. Any key to "
            "continue. Ctrl-C to exit."
        )
        try:
            input()
        except KeyboardInterrupt:
            logger.info("Exiting.")
            sys.exit(0)

    if not has_account:
        logger.warning(
            f"No account provided. Defaulting to {account}. Any key to continue. "
            "Ctrl-C to exit."
        )
        try:
            input()
        except KeyboardInterrupt:
            logger.info("Exiting.")
            sys.exit(0)

    user_slurm_params: str = " ".join(extra_slurm_args)
    extra_slurm_params: str = user_slurm_params
    # Check for at least nodes and ntasks, and if not provided, prompt the user to use defaults.
    nodes: int = 1
    if "nodes" not in extra_slurm_params:
        logger.warning(
            f"No nodes provided. Defaulting to {nodes}. Any key to continue. "
            "Ctrl-C to exit."
        )
        try:
            input()
            extra_slurm_params = f"{extra_slurm_params} --nodes={nodes}"
        except KeyboardInterrupt:
            logger.info("Exiting.")
            sys.exit(0)

    ntasks: int = 1
    if "ntasks" not in extra_slurm_params:
        logger.warning(
            f"No ntasks provided. Defaulting to {ntasks}. Any key to continue. "
            "Ctrl-C to exit."
        )
        try:
            input()
            extra_slurm_params = f"{extra_slurm_params} --ntasks={ntasks}"
        except KeyboardInterrupt:
            logger.info("Exiting.")
            sys.exit(0)

    workflows: List[Dict[str, str]] = []
    for wf_name in workflow_names:
        full_workflow_path: str = f"{lute_output_dir}/{wf_name}.dag"
        if not os.path.exists(full_workflow_path):
            included_wf_defn: str = f"{central_install}/workflows/common/{wf_name}.dag"
            if not os.path.exists(included_wf_defn):
                logger.error(
                    f"Workflow definition not found for workflow: {wf_name}. Skipping workflow."
                )
                continue
            shutil.copy(included_wf_defn, full_workflow_path)
        os.chmod(full_workflow_path, 0o666)

        param_string: str = (
            f"{launch_executable} -c {config_path} -W {full_workflow_path} --partition={partition} --account={account}"
        )
        if args.debug:
            param_string = f"{param_string} --debug"
        if args.test:
            param_string = f"{param_string} --test"

        # Update the DAG file in place with collected SLURM params
        config = DEFAULT_CONFIG.get(hutch.upper(), DEFAULT_CONFIG["ELSE"])
        update_dag_params(
            full_workflow_path,
            partition,
            account,
            extra_slurm_params,
            config,
            user_slurm_params,
        )

        # Build workflow dict with appropriate trigger
        if "smd" in wf_name:
            trigger = {"trigger": "START_OF_RUN"}
        else:
            trigger = {"trigger": "MANUAL"}

        workflows.append(
            {
                "name": f"lute_{wf_name}",
                "executable": arp_executable,
                "location": "S3DF",
                "parameters": param_string,
                **trigger,
            }
        )

    for workflow in workflows:
        logger.info(f"Creating eLog workflow for {workflow['name']}")
        krbticket: Any = KerberosTicket("HTTP@pswww.slac.stanford.edu")
        krbheaders: dict = krbticket.getAuthHeaders()
        url: str = (
            f"https://pswww.slac.stanford.edu/ws-kerb/lgbk/lgbk/{args.experiment}/ws"
            "/create_update_workflow_def"
        )
        post_params: Dict[str, Any] = {
            "url": url,
            "headers": krbheaders,
            "json": workflow,
        }
        resp: requests.models.Response = requests.post(**post_params)
        resp.raise_for_status()
        # Extra logging and such...


if __name__ == "__main__":
    main()
