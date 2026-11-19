"""Script to run functional tests of various LUTE workflows."""

__author__ = "Gabriel Dorlhiac"

import argparse
import logging
import os
import random
import shutil
import subprocess
import sys
import time
import uuid
import yaml
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
    overload,
)

import json
import requests
import socket
from requests.auth import HTTPBasicAuth
from requests.exceptions import HTTPError

from lute.execution.launch import (
    request_arp_token,
    retrieve_run_info,
    get_lute_launch_config,
    get_concurrent_job_steps,
    EnvLaunchInfo,
    LuteParams,
    LuteLaunchConfig,
)

# Requests, urllib have lots of debug statements. Only set level for this logger
logger: logging.Logger = logging.getLogger("Launch_Func_Tests")
handler: logging.Handler = logging.StreamHandler()
formatter: logging.Formatter = logging.Formatter(logging.BASIC_FORMAT)
handler.setFormatter(formatter)
logger.addHandler(handler)

if __debug__:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)


def _retrieve_prefect_creds_and_url(
    instance: str = "experimental",
) -> Tuple[str, str, str]:
    path: str = "/sdf/group/lcls/ds/tools/lute/prefect_{instance}.txt"
    if instance == "experimental":
        path = path.format(instance=instance)
    else:
        raise ValueError('`instance` must be "experimental"')
    with open(path, "r") as f:
        user_pw: str = f.readline().strip()
        url: str = f.readline().strip()
    user: str
    pw: str
    user, pw = user_pw.split(":")
    return user, pw, url


def _retrieve_airflow_pw(instance: str = "prod", is_admin: bool = False) -> str:
    user_type: str
    if is_admin:
        logger.debug("Running as operator.")
        user_type = "admin"
    else:
        logger.debug("Running as user.")
        user_type = "user"
    path: str = "/sdf/group/lcls/ds/tools/lute/airflow_{instance}_{user_type}.txt"
    if instance == "prod" or instance == "test":
        path = path.format(instance=instance, user_type=user_type)
    else:
        raise ValueError('`instance` must be either "test" or "prod"!')
    with open(path, "r") as f:
        pw: str = f.readline().strip()
    return pw


def modify_permissions(path: str, permissions: int) -> None:
    """Recursively set permissions for a path."""
    os.chmod(path, permissions)
    for root, dirs, files in os.walk(path):
        for d in dirs:
            os.chmod(os.path.join(root, d), permissions)

        for f in files:
            os.chmod(os.path.join(root, f), permissions)


@overload
def _run_subprocess_log(cmd: List[str], return_output: Literal[True]) -> str: ...


@overload
def _run_subprocess_log(cmd: List[str], return_output: Literal[False]) -> None: ...


@overload
def _run_subprocess_log(cmd: List[str]) -> None: ...


def _run_subprocess_log(cmd: List[str], return_output: bool = False) -> Optional[str]:
    """Run a subprocess with logging."""
    global logger

    out: str
    err: str
    out, err = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    ).communicate()
    if out:
        logger.info(out)
    if err:
        logger.info(err)

    if return_output:
        return out
    return None


def grep(match_str: str, in_file: str) -> List[str]:
    """Grep for specific lines of text output.

    Args:
        match_str (str): String to search for.

        in_file (str): File to search.

    Returns:
        lines (List[str]): The matches. It may be a list with just an empty
            string if nothing is found.
    """
    cmd: List[str] = ["grep", match_str, in_file]
    out: str = _run_subprocess_log(cmd, return_output=True)
    lines: List[str] = out.split("\n")
    return lines


def parse_yaml_value(input_config_yaml: str, key: str) -> str:
    """Retrieve a value for a specific YAML key.

    Currently only works for the header config section of a LUTE YAML file.
    E.g. for retrieving the experiment or run number.
    """
    lines: List[str] = grep(f"^{key}", input_config_yaml)
    key_val: str = lines[0]
    if "{{" in key_val:
        raise RuntimeError(
            "Substitutions not allowed in test configuration YAMLs! "
            f"Replace {key_val} with a defined value!"
        )
    val: str = key_val.split(":")[1].strip().replace('"', "")
    return val


def git_clone(repo: str, location: str, permissions: int) -> None:
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
        f"{location}/{repo_only}",
    ]
    _run_subprocess_log(cmd)
    modify_permissions(f"{location}/{repo_only}", permissions)


def inplace_sed(in_file: str, pattern: str) -> None:
    """Perform an in-place operation on a file using sed.

    Args:
        in_file (str): Path to the file to perform the substitution on.

        pattern (str): Operation. E.g. substitute with "s/old_text/new_text/g"
    """
    cmd: List[str] = ["sed", "-i", pattern, in_file]
    _run_subprocess_log(cmd)


def git_stash(path_to_repo: str) -> None:
    old_cwd: str = os.getcwd()
    os.chdir(path_to_repo)
    stash_cmd: List[str] = ["git", "stash"]
    _run_subprocess_log(stash_cmd)
    os.chdir(old_cwd)


def git_checkout_branch(path_to_repo: str, branch: str) -> None:
    # Not clear to me why need to stash on occassion??
    # But this seems to be required switching between some branches
    git_stash(path_to_repo)
    old_cwd: str = os.getcwd()
    os.chdir(path_to_repo)
    switch_cmd: List[str] = ["git", "switch", branch]
    _run_subprocess_log(switch_cmd)
    os.chdir(old_cwd)


def git_fetch_pr_branch(path_to_repo: str, github_id: int) -> None:
    old_cwd: str = os.getcwd()
    os.chdir(path_to_repo)
    new_pr_branch_name: str = "PR_BRANCH"
    fetch_cmd: List[str] = [
        "git",
        "fetch",
        "origin",
        f"pull/{github_id}/head:{new_pr_branch_name}",
    ]
    _run_subprocess_log(fetch_cmd)
    os.chdir(old_cwd)
    git_checkout_branch(path_to_repo, new_pr_branch_name)


def run_workflow_airflow(
    lute_location: str,
    config_file: str,
    workflow_file: str,
    use_test_inst: bool = False,
    is_admin: bool = False,
) -> bool:
    """Run a workflow.

    Args:
        lute_location (str): Path to the LUTE installation.

        config_file (str): Path to the configuration YAML.

        workflow_file (str): Path to the DAG definition YAML.

        use_test_inst (bool): Whether to use the test Airflow instance (instead
            of production instance). Default: False.

        is_admin (bool): Whether running as administrator account. Default: False.

    Returns:
        is_successful (bool): True if workflow returns successful. False otherwise.
    """
    airflow_instance: str
    instance_str: str
    if use_test_inst:
        airflow_instance = "http://172.24.5.190:8080"
        instance_str = "test"
    else:
        airflow_instance = "http://172.24.5.247:8080"
        instance_str = "prod"

    wf_name: str = "test_dynamic"
    airflow_api_endpoints: Dict[str, str] = {
        "health": "api/v1/health",
        "run_dag": f"api/v1/dags/lute_{wf_name}/dagRuns",
        "get_tasks": f"api/v1/dags/lute_{wf_name}/tasks",
        "get_xcom": (  # Need to format dag_run_id, task_id, xcom_key
            f"api/v1/dags/lute_{wf_name}/dagRuns/{{dag_run_id}}/taskInstances"
            f"/{{task_id}}/xcomEntries/{{xcom_key}}"
        ),
        # Only for User-Specified workflows
        "mod_dag": f"api/v1/dags/lute_{wf_name}",  # Delete, pause/unpause, etc.
        "create_defn": "api/v1/variables",
        "update_defn": "api/v1/variables/user_workflow",
        "parse_file": "api/v1/parseDagFile/{file_token}",
    }

    pw: str = _retrieve_airflow_pw(instance_str, is_admin=is_admin)
    user_name: str = "btx" if is_admin else "lcls_user"
    auth: HTTPBasicAuth = HTTPBasicAuth(user_name, pw)
    resp: requests.models.Response = requests.get(
        f"{airflow_instance}/{airflow_api_endpoints['health']}",
        auth=auth,
    )
    resp.raise_for_status()

    params: Dict[str, Union[str, bool]] = {
        "config_file": config_file,
        "debug": True,
    }

    wf_defn: Dict[str, Any] = {}

    if not os.path.exists(workflow_file):
        logger.error("Workflow definition path does not exist! Exiting!")
        sys.exit(-1)
    with open(workflow_file, "r") as f:
        wf_defn = yaml.load(f, yaml.FullLoader)

    # Update user workflow definition in Airflow
    new_workflow: Dict[str, str] = {
        "key": "user_workflow",
        "value": json.dumps(wf_defn),
    }
    resp = requests.patch(
        f"{airflow_instance}/{airflow_api_endpoints['update_defn']}",
        json=new_workflow,
        auth=auth,
    )
    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError as err:
        if err.response.status_code == 404:
            # Workflow definition not found so previous DAG completed properly
            resp = requests.post(
                f"{airflow_instance}/{airflow_api_endpoints['create_defn']}",
                json=new_workflow,
                auth=auth,
            )
            resp.raise_for_status()
        else:
            raise
    # Let's wait some time and update again...
    # Try and allow Airflow to get used to the Dynamic DAG it doesn't like...
    time.sleep(2)
    resp = requests.patch(
        f"{airflow_instance}/{airflow_api_endpoints['update_defn']}",
        json=new_workflow,
        auth=auth,
    )
    resp.raise_for_status()
    logger.debug("Sent new workflow definition.")
    resp = requests.get(
        f"{airflow_instance}/{airflow_api_endpoints['mod_dag']}",
        auth=auth,
    )
    resp.raise_for_status()
    file_token: str = resp.json()["file_token"]
    f_endpoint: str = airflow_api_endpoints["parse_file"].format(file_token=file_token)
    resp = requests.put(f"{airflow_instance}/{f_endpoint}", auth=auth)
    resp.raise_for_status()
    logger.debug("Re-parsed DAG for setup with new workflow.")

    # Experiment, run #, and ARP env variables come from ARP submission only
    # We override above or exit if we cannot, so we cast here
    experiment: Optional[str] = os.getenv("EXPERIMENT")
    run_num: Optional[str] = os.getenv("RUN_NUM")
    arp_job_id: Optional[str] = os.getenv("ARP_JOB_ID")
    jid_authorization: Optional[str] = os.getenv("Authorization")
    assert isinstance(experiment, str)
    assert isinstance(run_num, str)
    assert isinstance(arp_job_id, str)
    assert isinstance(jid_authorization, str)

    run_type, is_daq2 = retrieve_run_info(experiment, run_num, jid_authorization)

    launch_info: EnvLaunchInfo = {
        "experiment": experiment,
        "run_num": run_num,
        "authorization": jid_authorization,
        "arp_job_id": arp_job_id,
        "kerb_file": cache_file,
    }

    conf: LuteLaunchConfig = get_lute_launch_config(
        launch_info=launch_info,
        run_type=run_type,
        is_daq2=is_daq2,
        lute_params=params,
        slurm_params=extra_args,
        workflow_defn=wf_defn,
        lute_location=lute_location,
        executable_subdir="launch_scripts",  # Airflow script location
    )

    dag_run_data: Dict[str, Any] = {
        "dag_run_id": str(uuid.uuid4()),
        "conf": conf,
    }

    # Get Task information
    task_ids: List[str]
    # Airflow shouldn't have list of Tasks yet so we parse manually
    task_ids = ["retrieve_workflow"]

    def get_names(wf_defn: Dict[str, Any], names: List[str]) -> None:
        names.append(f"user_workflow.{wf_defn['task_name']}")
        for wf_new in wf_defn["next"]:
            get_names(wf_new, names)

    get_names(wf_defn, task_ids)
    task_ids = sorted(task_ids)
    task_id_str: str = ",\n\t- ".join(tid for tid in task_ids)
    logger.info(
        f"Contains Managed Tasks (alphabetical, not execution order):\n\t- {task_id_str}"
    )

    # Submit hopefully Airflow will do okay
    dag_run_data["dag_run_id"] = str(uuid.uuid4())
    resp = requests.post(
        f"{airflow_instance}/{airflow_api_endpoints['run_dag']}",
        json=dag_run_data,
        auth=auth,
    )
    resp.raise_for_status()
    dag_run_id: str = dag_run_data["dag_run_id"]
    logger.info(f"Submitted DAG (Workflow): {wf_name}\nDAG_RUN_ID: {dag_run_id}")
    dag_state: str = resp.json()["state"]
    logger.info(f"DAG is {dag_state}")
    # Enter loop for checking status
    time.sleep(1)
    # Same as run_dag endpoint, but needs to include the dag_run_id on the end
    url: str = f"{airflow_instance}/{airflow_api_endpoints['run_dag']}/{dag_run_id}"
    # Pulling logs for each Task via XCom
    xcom_key: str = "log"
    completed_tasks: Dict[str, str] = {}  # Remember exit status of each Task
    logged_running: List[str] = []  # Keep track to only print "running" once
    while True:
        time.sleep(1)
        # DAG Status
        resp = requests.get(url, auth=auth)
        resp.raise_for_status()
        dag_state = resp.json()["state"]
        # Check Task instances
        task_url: str = f"{url}/taskInstances"
        resp = requests.get(task_url, auth=auth)
        resp.raise_for_status()
        instance_information: List[Dict[str, Any]] = resp.json()["task_instances"]
        for inst in instance_information:
            task_id: str = inst["task_id"]
            task_state: Optional[str] = inst["state"]
            if task_id not in completed_tasks and task_state not in (
                None,
                "scheduled",
                "queued",
                "removed",
            ):
                if task_id not in logged_running:
                    # Should be "running" by first time it reaches here.
                    # Or e.g. "upstream_failed"... Setup to skip "scheduled"
                    logger.info(f"{task_id} state: {task_state}")
                    logged_running.append(task_id)

                if task_state in ("success", "failed"):
                    # Only pushed to XCOM at the end of each Task
                    xcom_url: str = (
                        f"{airflow_instance}/{airflow_api_endpoints['get_xcom']}"
                    )
                    xcom_url = xcom_url.format(
                        dag_run_id=dag_run_id,
                        task_id=task_id,
                        xcom_key=xcom_key,
                    )
                    try:
                        resp = requests.get(xcom_url, auth=auth)
                        resp.raise_for_status()
                        logs: str = resp.json()["value"]  # Only want to print once.
                        logger.info(f"Providing logs for {task_id}")
                        print("-" * 50, flush=True)
                        print(logs, flush=True)
                        print("-" * 50, flush=True)
                    except HTTPError:
                        # retrieve_workflow has no logs...
                        logger.info(f"No logs for {task_id}.")
                    logger.info(f"End of logs for {task_id}")
                    completed_tasks[task_id] = task_state

                # Ignore type check since cannot be None here
                elif task_state in ("upstream_failed"):  # type: ignore
                    # upstream_failed never launches so has no log
                    completed_tasks[task_id] = task_state  # type: ignore

        if dag_state in ("queued", "running"):
            continue
        logger.info(f"DAG exited: {dag_state}")
        break
    if dag_state == "failed":
        return False
    else:
        return True


def run_workflow_prefect(
    lute_location: str,
    config_file: str,
    workflow_file: str,
) -> bool:
    """Run a workflow.

    Args:
        lute_location (str): Path to the LUTE installation.

        config_file (str): Path to the configuration YAML.

        workflow_file (str): Path to the DAG definition YAML.

    Returns:
        is_successful (bool): True if workflow returns successful. False otherwise.
    """

    flow_name: str = "lute_dynamic"
    deployment_name: str = "dev"

    user: str
    pw: str
    PREFECT_API_URL: str
    user, pw, PREFECT_API_URL = _retrieve_prefect_creds_and_url()
    auth: HTTPBasicAuth = HTTPBasicAuth(user, pw)

    csrf_endpoint: str = f"{PREFECT_API_URL}/csrf-token"
    name_endpoint: str = (
        f"{PREFECT_API_URL}/deployments/name/{flow_name}/{deployment_name}"
    )

    if not os.path.exists(workflow_file):
        logger.error("Workflow definition path does not exist! Exiting!")
        sys.exit(-1)

    wf_defn: Dict[str, Any] = {}
    with open(workflow_file, "r") as f:
        wf_defn = yaml.load(f, yaml.FullLoader)

    # Experiment, run #, and ARP env variables come from ARP submission only
    # We override above or exit if we cannot, so we cast here
    experiment: Optional[str] = os.getenv("EXPERIMENT")
    run_num: Optional[str] = os.getenv("RUN_NUM")
    arp_job_id: Optional[str] = os.getenv("ARP_JOB_ID")
    jid_authorization: Optional[str] = os.getenv("Authorization")
    assert isinstance(experiment, str)
    assert isinstance(run_num, str)
    assert isinstance(arp_job_id, str)
    assert isinstance(jid_authorization, str)

    run_type, is_daq2 = retrieve_run_info(experiment, run_num, jid_authorization)

    params: LuteParams = {
        "config_file": config_file,
        "debug": True,
    }

    launch_info: EnvLaunchInfo = {
        "experiment": experiment,
        "run_num": run_num,
        "authorization": jid_authorization,
        "arp_job_id": arp_job_id,
        "kerb_file": cache_file,
    }

    conf: LuteLaunchConfig = get_lute_launch_config(
        launch_info=launch_info,
        run_type=run_type,
        is_daq2=is_daq2,
        lute_params=params,
        slurm_params=extra_args,
        workflow_defn=wf_defn,
        lute_location=lute_location,
        executable_subdir="launch_scripts",
    )

    # Get CSRF
    ##############################################
    resp = requests.get(csrf_endpoint, auth=auth, params={"client": user})

    token: str = resp.json()["token"]
    client: str = resp.json()["client"]

    # Get ID from name
    ##############################################
    resp = requests.get(name_endpoint, auth=auth)

    deployment_id: str = resp.json()["id"]

    # Launch flow_run
    ##############################################
    launch_endpoint: str = (
        f"{PREFECT_API_URL}/deployments/{deployment_id}/create_flow_run"
    )

    data: Dict[str, Any] = {"parameters": {"flow_conf": conf}}
    headers: Dict[str, str] = {
        "Prefect-Csrf-Token": token,
        "Prefect-Csrf-Client": client,
    }

    resp = requests.post(launch_endpoint, headers=headers, json=data, auth=auth)

    flow_run_id: str = resp.json()["id"]

    # Run loop, gather logs, etc.
    ##############################################
    flow_run_state_endpoint: str = f"{PREFECT_API_URL}/flow_runs/{flow_run_id}"
    log_endpoint: str = f"{PREFECT_API_URL}/logs/filter"

    log_payload: Dict[str, Any] = {
        "logs": {
            "level": {
                "ge_": 20,
            },
            "flow_run_id": {"any_": [flow_run_id]},
        },
        "sort": "TIMESTAMP_ASC",
    }

    resp = requests.get(flow_run_state_endpoint, auth=auth)
    state: str = resp.json()["state_type"]

    last_log_idx: int = 0
    while state in ("SCHEDULED", "PENDING", "RUNNING"):
        time.sleep(5)
        # Retrieve logs for flow run, printing new ones... Bit wasteful...
        resp = requests.post(log_endpoint, headers=headers, auth=auth, json=log_payload)
        current_len_logs: int = len(resp.json())
        log_dict: Dict[str, Any]
        for log_dict in resp.json()[last_log_idx:current_len_logs]:
            print(log_dict["message"])
        last_log_idx = current_len_logs

        resp = requests.get(flow_run_state_endpoint, auth=auth)
        state = resp.json()["state_type"]

    if state in ("CANCELLED", "FAILED", "CRASHED"):
        return False
    else:
        return True


def find_random_port(
    min_port: int = 41923, max_port: int = 64324, max_tries: int = 20
) -> Optional[int]:
    ports: List[int] = random.choices(range(min_port, max_port), k=max_tries)

    for port in ports:
        sock: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            sock.bind(("", port))
            sock.close()
            del sock
            return port
        except Exception:
            continue
    return None


def run_workflow_maestro(
    lute_location: str,
    config_file: str,
    workflow_file: str,
    extra_args: List[str],
) -> bool:
    """Run a workflow using Maestro.

    Args:
        lute_location (str): Path to the LUTE installation.

        config_file (str): Path to the configuration YAML.

        workflow_file (str): Path to the DAG definition YAML.

        extra_args (List[str]): The set of additional arguments which should be
            for SLURM.

    Returns:
        is_successful (bool): True if workflow returns successful. False otherwise.
    """
    from maestro._maestro import _maestro
    from maestro.parser import load_lute_dag

    # Experiment, run #, and ARP env variables come from ARP submission only
    # BUT, we should've added them to the env by the time this is called
    experiment: str = cast(str, os.getenv("EXPERIMENT"))
    run_num: str = cast(str, os.getenv("RUN_NUM"))
    jid_authorization: str = cast(str, os.getenv("Authorization"))

    run_type, is_daq2 = retrieve_run_info(experiment, run_num, jid_authorization)

    wf_defn: List[_maestro.JobStep] = load_lute_dag(
        workflow_path=workflow_file,
        lute_location=lute_location,
        executable_subdir="install/bin",
        config_file=config_file,
        debug=True,
        default_slurm_params=" ".join(extra_args),
    )

    num_concurrent_steps: int = get_concurrent_job_steps(wf_defn)
    manager_host: str = socket.gethostname()
    manager_port: Optional[int] = find_random_port()
    if manager_port is None:
        raise RuntimeError("Cannot bind a port!")
    os.environ["LUTE_MANAGER_URL"] = f"{manager_host}:{manager_port}"

    # fmt: off
    manager_params: _maestro.ManagerParameters = _maestro.ManagerParameters(
        num_concurrent_steps,                     # Manager threads
        2,                                        # Server threads
        True,                                     # Unbuffered logs
        "0.0.0.0",                                # Server IP
        manager_port,                             # Server port
        _maestro.LauncherType.SlurmLauncherType,  # Launch mechanism
        is_daq2 if is_daq2 is not None else True, # Is daq2?
        run_type,                                 # Run type
    )
    # fmt: on

    status: str = _maestro.run_workflow(wf_defn, manager_params)
    if status == "FAILED":
        logger.error(f"Maestro workflow failed: {status}")
        return False

    logger.info(f"Maestro workflow exited: {status}")
    return True


def clean_up(
    cache_file: Optional[str], lute_location: Optional[str], output_location: str
) -> None:
    # We had to do some funny business to get Kerberos credentials...
    # Cleanup now that we're done
    logger.debug("Removing duplicate Kerberos credentials.")
    # This should be defined if we get here
    # Format is FILE:/.../...
    if cache_file is not None:
        try:
            os.remove(cache_file[5:])
            os.rmdir(f"{os.path.expanduser('~')}/.tmp_cache")
        except FileNotFoundError:
            logger.error("No cache file found to remove.")

    # None when running from the caller's own checkout (--no_clone): keep it
    if lute_location is not None:
        logger.info(f"Cleaning up {lute_location}")
        shutil.rmtree(lute_location)
    logger.info(f"Cleaning up {output_location}")
    shutil.rmtree(output_location)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="functional_airflow_test_suite",
        description="Run the LUTE functional test suite.",
        epilog="Refer to https://github.com/slac-lcls/lute for more information.",
    )
    # Airflow vs Prefect arguments
    parser.add_argument(
        "--use_prefect",
        help="Use prefect (experimental) instead of maestro.",
        action="store_true",
    )
    parser.add_argument(
        "--use_airflow",
        help="Use Airflow instead of maestro.",
        action="store_true",
    )
    parser.add_argument(
        "-a",
        "--admin",
        help="Run as Airflow admin. Requires permissions. Ignored if using prefect.",
        action="store_true",
    )
    parser.add_argument(
        "--test_airflow",
        help="Use test Airflow instance. Ignored if using prefect.",
        action="store_true",
    )
    # Options for running specific versions of LUTE
    parser.add_argument(
        "--no_clone",
        help="If passed, will use the build of LUTE that the script was called from.",
        action="store_true",
    )
    parser.add_argument(
        "--git_pr_id",
        help="Check out a specific GitHub PR ID of LUTE to run (a PR branch).",
        type=int,
    )
    parser.add_argument(
        "--git_tag",
        help="Check out a specific git tag of LUTE to run (e.g. a release).",
        type=str,
    )
    parser.add_argument(
        "-r", "--run_dir", help="Directory to install LUTE to.", type=str, required=True
    )
    # Choice of using specific tests, and whether to delete output files.
    parser.add_argument(
        "--no_delete",
        help="If passed, do not delete output files when tests are finished.",
        action="store_true",
    )
    parser.add_argument(
        "--run_tests",
        help=(
            "Provide a comma-separated string of tests to run.  If provided, this "
            "script will only run those, rather than the default behaviour of "
            "running all tests. E.g: --run_these_tests test2,test5. Tests that do not "
            "exist are silently ignored."
        ),
    )
    parser.add_argument(
        "--tests_dir",
        help=(
            "Specify an alternative path to tests than those from the LUTE clone.\n"
            "Must have the same directory structure: $DIR/test1/... $DIR/test2/...\n"
            "If this flag and --use_local_tests are both passed, this one is used."
        ),
        type=str,
    )
    parser.add_argument(
        "--use_local_tests",
        help=(
            "Use the tests from the installation of LUTE where this script is called,\n"
            "rather than those from the clone of LUTE which is run against, or another\n"
            "directory if passed. If this flag and --tests_dir are both passed, "
            "--tests_dir is used."
        ),
        action="store_true",
    )
    # Whether to allow output location to exist or not
    # To avoid accidental overwrites, by default we exit if the output location
    # already existed prior to running tests
    parser.add_argument(
        "--exist_ok",
        help=(
            "If passed, tests will continue even if the determined output directory "
            "already existed. It is possible things may be overwritten in this case."
        ),
        action="store_true",
    )

    args: argparse.Namespace
    extra_args: List[str]
    args, extra_args = parser.parse_known_args()

    cache_file: Optional[str] = os.getenv("KRB5CCNAME")
    if cache_file is None:
        logger.error("No Kerberos cache. Try running `kinit` and resubmitting.")
        sys.exit(-1)

    run_dir: str
    lute_location: str
    if not args.no_clone:
        run_dir = args.run_dir
        logger.debug(f"Cloning LUTE to {run_dir}")
        git_clone("slac-lcls/lute", run_dir, 0o777)
        if args.git_tag is not None and args.git_pr_id is not None:
            logger.warning(
                "Provided both a git tag and git ID to use. Will default to using the ID."
            )
            logger.info(f"Switching to PR branch ID {args.git_pr_id}")
            git_fetch_pr_branch(f"{run_dir}/lute", args.git_pr_id)
        elif args.git_tag is not None:
            logger.info(f"Switching to tag {args.git_tag}")
            git_checkout_branch(f"{run_dir}/lute", args.git_tag)
        elif args.git_pr_id is not None:
            logger.info(f"Switching to PR branch ID {args.git_pr_id}")
            git_fetch_pr_branch(f"{run_dir}/lute", args.git_pr_id)
        else:
            logger.info("Running LUTE from dev branch.")
        lute_location = f"{run_dir}/lute"
    else:
        run_dir = os.path.abspath(f"{os.path.dirname(__file__)}/../..")
        lute_location = os.path.abspath(f"{os.path.dirname(__file__)}/..")

    clone_location: Optional[str] = None if args.no_clone else lute_location
    output_location: str = f"{run_dir}/lute_output"
    logger.info(f"Will write output to {output_location}")
    if not os.path.exists(output_location):
        os.makedirs(output_location, mode=0o777)
        os.chmod(output_location, mode=0o777)
    else:
        if not args.exist_ok:
            logger.error(
                f"Output location ({output_location}) already exists! Exiting now. "
                "If you intended this, you can resubmit with the `--exist_ok` flag."
            )
            sys.exit(-1)
        logger.warning(f"Output location ({output_location}) already exists.")

    func_tests_dir: str
    if args.tests_dir is not None:
        if args.use_local_tests:
            logger.warning(
                "Provided both `--tests_dir` and `--use_local_tests`. Will use `--tests_dir`."
            )
        func_tests_dir = args.tests_dir
    elif args.use_local_tests:
        func_tests_dir = f"{os.path.dirname(__file__)}/functional"
    else:
        func_tests_dir = f"{lute_location}/tests/functional"

    def use_test(
        test_dir: str, func_tests_dir: str, usable_tests_str: Optional[str] = None
    ) -> bool:
        if usable_tests_str is not None:
            usable_dirs: List[str] = [
                f"{func_tests_dir}/{d}" for d in usable_tests_str.split(",")
            ]
            return test_dir in usable_dirs
        else:
            return test_dir != func_tests_dir

    test_dirs: List[str] = [
        x[0]
        for x in os.walk(func_tests_dir)
        if use_test(x[0], func_tests_dir, args.run_tests)
    ]
    logger.info(f"Will attempt running {len(test_dirs)} tests")

    num_successful: int = 0
    num_unsuccessful: int = 0
    try:
        for test_dir in test_dirs:
            test_name: str = test_dir.split("/")[-1]
            logger.info(f"Running test: {test_name}")
            # We assume that each test directory has a config.yaml and dag.yaml
            config_file: str = f"{test_dir}/config.yaml"
            sed_pattern: str = f's|work_dir:\(.*\)|work_dir: \\"{output_location}\\"|g'
            inplace_sed(config_file, sed_pattern)
            wf_file: str = f"{test_dir}/dag.yaml"
            should_fail: bool = (
                True if os.path.exists(f"{test_dir}/SHOULD_FAIL") else False
            )

            # Replace DAG SLURM account for account provided on command-line
            for arg in extra_args:
                if "--account=" in arg:
                    sed_pattern = f"s|--account=[^[:space:]]*|{arg}|g"
                    inplace_sed(wf_file, sed_pattern)
                elif "--constraint=" in arg:
                    sed_pattern = f"s|slurm_params: \\([\"']\\)|slurm_params: \\1{arg} |g"
                    inplace_sed(wf_file, sed_pattern)

            # Retrieve the experiment and run from each test YAML
            experiment: str = parse_yaml_value(config_file, "experiment")
            run: str = parse_yaml_value(config_file, "run")

            # Setup environment variables -> These are passed to airflow (or prefect)
            # run_workflow function uses them
            os.environ["EXPERIMENT"] = experiment
            os.environ["RUN_NUM"] = run
            os.environ["Authorization"] = request_arp_token(experiment)
            os.environ["ARP_JOB_ID"] = str(uuid.uuid4())

            run_workflow: Union[
                Callable[[str, str, str, bool, bool], bool],
                Callable[[str, str, str], bool],
                Callable[[str, str, str, List[str]], bool],
            ]
            is_successful: bool
            if args.use_airflow:
                run_workflow = run_workflow_airflow
                is_successful = run_workflow(
                    lute_location=lute_location,
                    config_file=config_file,
                    workflow_file=wf_file,
                    use_test_inst=args.test_airflow,
                    is_admin=args.admin,
                )
            elif args.use_prefect:
                run_workflow = run_workflow_prefect
                is_successful = run_workflow(
                    lute_location=lute_location,
                    config_file=config_file,
                    workflow_file=wf_file,
                )
            else:
                run_workflow = run_workflow_maestro
                is_successful = run_workflow(
                    lute_location=lute_location,
                    config_file=config_file,
                    workflow_file=wf_file,
                    extra_args=extra_args,
                )

            if is_successful:
                if should_fail:
                    num_unsuccessful += 1
                    logger.error(
                        f"Test workflow {test_name} completed successfully but should fail!"
                    )
                else:
                    num_successful += 1
                    logger.info(f"Test workflow {test_name} completed successfully.")
            else:
                if should_fail:
                    num_successful += 1
                    logger.info(
                        f"Test workflow {test_name} was unsuccessful but this is marked as intentional."
                    )
                else:
                    num_unsuccessful += 1
                    logger.error(f"Test workflow {test_name} was unsuccessfull!")
    except Exception as e:
        logger.error(f"Error in testing framework: {e}")
        if not args.no_delete:
            clean_up(cache_file, clone_location, output_location)
        sys.exit(-1)

    logger.info(f"Ran {len(test_dirs)} tests. {num_successful} were successful.")
    if not args.no_delete:
        clean_up(cache_file, clone_location, output_location)
