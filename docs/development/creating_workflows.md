# Workflows with Airflow
**Note:** Airflow uses the term **DAG**, or directed acyclic graph, to describe workflows of tasks with defined (and acyclic) connectivities. This page will use the terms workflow and DAG interchangeably.

## Relevant Components
In addition to the core LUTE package, a number of components are generally involved to run a workflow. The current set of scripts and objects are used to interface with Airflow, and the SLURM job scheduler. The core LUTE library can also be used to run workflows using different backends, and in the future these may be supported.

For building and running workflows using SLURM and Airflow, the following components are necessary, and will be described in more detail below:
- Airflow launch script: `launch_airflow.py`
  - This has a wrapper batch submission script: `submit_launch_airflow.sh` . When running using the ARP (from the eLog), you **MUST** use this wrapper script instead of the Python script directly.
- SLURM submission script: `submit_slurm.sh`
- Airflow operators:
  - `JIDSlurmOperator`

## Launch/Submission Scripts
## `launch_airflow.py`
Sends a request to an Airflow instance to submit a specific DAG (workflow). This script prepares an HTTP request with the appropriate parameters in a specific format.

A request involves the following information, most of which is retrieved automatically:
```py
dag_run_data: Dict[str, Union[str, Dict[str, Union[str, int, List[str]]]]] = {
    "dag_run_id": str(uuid.uuid4()),
    "conf": {
        "experiment": os.environ.get("EXPERIMENT"),
        "run_id": f"{os.environ.get('RUN_NUM')}{datetime.datetime.utcnow().isoformat()}",
        "JID_UPDATE_COUNTERS": os.environ.get("JID_UPDATE_COUNTERS"),
        "ARP_ROOT_JOB_ID": os.environ.get("ARP_JOB_ID"),
        "ARP_LOCATION": os.environ.get("ARP_LOCATION", "S3DF"),
        "Authorization": os.environ.get("Authorization"),
        "user": getpass.getuser(),
        "lute_params": params,
        "slurm_params": extra_args,
        "workflow": wf_defn,  # Used only for custom DAGs. See below under advanced usage.
    },
}
```
Note that the environment variables are used to fill in the appropriate information because this script is intended to be launched primarily from the ARP (which passes these variables). The ARP allows for the launch job to be defined in the experiment eLog and submitted automatically for each new DAQ run. The environment variables `EXPERIMENT` and `RUN` can alternatively be defined prior to submitting the script on the command-line.

The script takes a number of parameters:

```bash
launch_airflow.py -c <path_to_config_yaml> -w <workflow_name> [--debug] [--test] [-e <exp>] [-r <run>] [SLURM_ARGS]
```

- `-c` refers to the path of the configuration YAML that contains the parameters for each **managed** `Task` in the requested workflow.
- `-w` is the name of the DAG (workflow) to run. By convention each DAG is named by the Python file it is defined in. (See below).
  - **NOTE:** For advanced usage, a custom DAG can be provided at **run** time using `-W` (capital W) followed by the path to the workflow instead of `-w`. See below for further discussion on this use case.
- `--debug` is an optional flag to run all steps of the workflow in debug mode for verbose logging and output.
- `--test` is an optional flag which will use the test Airflow instance. By default the script will make requests of the standard production Airflow instance.
- `-e` is used to pass the experiment name. Needed if not using the ARP, i.e. running from the command-line.
- `-r` is used to pass a run number. Needed if not using the ARP, i.e. running from the command-line.
- `SLURM_ARGS` are SLURM arguments to be passed to the `submit_slurm.sh` script which are used for each individual **managed** `Task`. These arguments to do NOT affect the submission parameters for the job running `launch_airflow.py` (if using `submit_launch_airflow.sh` below).


**Lifetime**
This script will run for the entire duration of the **workflow (DAG)**. After making the initial request of Airflow to launch the DAG, it will enter a status update loop which will keep track of each individual job (each job runs one managed `Task`)  submitted by Airflow. At the end of each job it will collect the log file, in addition to providing a few other status updates/debugging messages, and append it to its own log. This allows all logging for the entire workflow (DAG) to be inspected from an individual file. This is particularly useful when running via the eLog, because only a single log file is displayed.

### `submit_launch_airflow.sh`
This script is only necessary when running from the eLog using the ARP. The initial job submitted by the ARP can not have a duration of longer than 30 seconds, as it will then time out. As the `launch_airflow.py` job will live for the entire duration of the workflow, which is often much longer than 30 seconds, the solution was to have a wrapper which submits the `launch_airflow.py` script to run on the S3DF batch nodes. Usage of this script is mostly identical to `launch_airflow.py`. All the arguments are passed transparently to the underlying Python script with the exception of the first argument which **must** be the location of the underlying `launch_airflow.py` script. The wrapper will simply launch a batch job using minimal resources (1 core). While the primary purpose of the script is to allow running from the eLog, it is also an useful wrapper generally, to be able to submit the previous script as a SLURM job.

Usage:

```bash
submit_launch_airflow.sh /path/to/launch_airflow.py -c <path_to_config_yaml> -w <workflow_name> [--debug] [--test] [-e <exp>] [-r <run>] [SLURM_ARGS]
```

## `submit_slurm.sh`
Launches a job on the S3DF batch nodes using the SLURM job scheduler. This script launches a single **managed** `Task` at a time. The usage is as follows:
```bash
submit_slurm.sh -c <path_to_config_yaml> -t <MANAGED_task_name> [--debug] [SLURM_ARGS ...]
```
As a reminder the **managed** `Task` refers to the `Executor`-`Task` combination. The script does not parse any SLURM specific parameters, and instead passes them transparently to SLURM. At least the following two SLURM arguments must be provided:
```bash
--partition=<...> # Usually partition=milano
--account=<...> # Usually account=lcls:$EXPERIMENT
```
Generally, resource requests will also be included, such as the number of cores to use. A complete call may look like the following:
```bash
submit_slurm.sh -c /sdf/data/lcls/ds/hutch/experiment/scratch/config.yaml -t Tester --partition=milano --account=lcls:experiment --ntasks=100 [...]
```

When running a workflow using the `launch_airflow.py` script, each step of the workflow will be submitted using this script.

## Operators
`Operator`s are the objects submitted as individual steps of a DAG by Airflow. They are conceptually linked to the idea of a task in that each task of a workflow is generally an operator. Care should be taken, not to confuse them with LUTE `Task`s or **managed** `Task`s though. There is, however, usually a one-to-one correspondance between a `Task` and an `Operator`.

Airflow runs on a K8S cluster which has no access to the experiment data. When we ask Airflow to run a DAG, it will launch an `Operator` for each step of the DAG. However, the `Operator` itself cannot perform productive analysis without access to the data. The solution employed by `LUTE` is to have a limited set of `Operator`s which do not perform analysis, but instead request that a `LUTE` **managed** `Task`s be submitted on the batch nodes where it can access the data. There may be small differences between how the various provided `Operator`s do this, but in general they will all make a request to the **job interface daemon** (JID) that a new SLURM job be scheduled using the `submit_slurm.sh` script described above.

Therefore, running a typical Airflow DAG involves the following steps:

1. `launch_airflow.py` script is submitted, usually from a definition in the eLog.
2. The `launch_airflow` script requests that Airflow run a specific DAG.
3. The Airflow instance begins submitting the `Operator`s that makeup the DAG definition.
4. Each `Operator` sends a request to the `JID` to submit a job.
5. The `JID` submits the `elog_submit.sh` script with the appropriate **managed** `Task`.
6. The **managed** `Task` runs on the batch nodes, while the `Operator`, requesting updates from the JID on job status, waits for it to complete.
7. Once a **managed** `Task` completes, the `Operator` will receieve this information and tell the Airflow server whether the job completed successfully or resulted in failure.
8. The Airflow server will then launch the next step of the DAG, and so on, until every step has been executed.

Currently, the following `Operator`s are maintained:
- `JIDSlurmOperator`: The standard `Operator`. Each instance has a one-to-one correspondance with a LUTE **managed** `Task`.

### `JIDSlurmOperator` arguments
- `task_id`: This is nominally the name of the task on the Airflow side. However, for simplicity this is used 1-1 to match the name of a **managed** Task defined in LUTE's `managed_tasks.py` module. I.e., it should the name of an `Executor("Task")` object which will run the specific Task of interest. This **must** match the name of a defined managed Task.
- `max_cores`: Used to cap the maximum number of cores which should be requested of SLURM. By default all jobs will run with the same number of cores, which should be specified when running the `launch_airflow.py` script (either from the ARP, or by hand). This behaviour was chosen because in general we want to increase or decrease the core-count for all `Task`s uniformly, and we don't want to have to specify core number arguments for each job individually. Nonetheless, on occassion it may be necessary to cap the number of cores a specific job will use. E.g. if the default value specified when launching the Airflow DAG is multiple cores, and one job is single threaded, the core count can be capped for that single job to 1, while the rest run with multiple cores.
- `max_nodes`: Similar to the above. This will make sure the `Task` is distributed across no more than a maximum number of nodes. This feature is useful for, e.g., multi-threaded software which does not make use of tools like `MPI`. So, the `Task` can run on multiple cores, but only within a single node.
- `require_partition`: This option is a string that forces the use of a specific S3DF partition for the **managed** `Task` submitted by the Operator. E.g. typically a LCLS user will use `--partition=milano` for CPU-based workflows; however, if a specific `Task` requires a GPU you may use `JIDSlurmOperator("MyTaskRunner", require_partition="ampere")` to override the partition for that single `Task`.
- `custom_slurm_params`: You can provide a string of parameters which will be used in its entirety to replace any and all default arguments passed by the launch script. This method is not recommended for general use and is mostly used for dynamic DAGs described at the end of the document.


# Creating a new workflow
Defining a new workflow involves creating a **new** module (Python file) in the directory `workflows/airflow`, creating a number of `Operator` instances within the module, and then drawing the connectivity between them. At the top of the file an Airflow DAG is created and given a name. By convention all `LUTE` workflows use the name of the file as the name of the DAG. The following code can be copied exactly into the file:

```py
from datetime import datetime
import os
from airflow import DAG
from lute.operators.jidoperators import JIDSlurmOperator # Import other operators if needed

dag_id: str = f"lute_{os.path.splitext(os.path.basename(__file__))[0]}"
description: str = (
    "Run SFX processing using PyAlgos peak finding and experimental phasing"
)

dag: DAG = DAG(
    dag_id=dag_id,
    start_date=datetime(2024, 3, 18),
    schedule_interval=None,
    description=description,
)
```

Once the DAG has been created, a number of `Operator`s must be created to run the various LUTE analysis operations. As an example consider a partial SFX processing workflow which includes steps for peak finding, indexing, merging, and calculating figures of merit. Each of the 4 steps will have an `Operator` instance which will launch a corresponding `LUTE` **managed** `Task`, for example:

```py
# Using only the JIDSlurmOperator
# syntax: JIDSlurmOperator(task_id="LuteManagedTaskName", dag=dag) # optionally, max_cores=123)
peak_finder: JIDSlurmOperator = JIDSlurmOperator(task_id="PeakFinderPyAlgos", dag=dag)

# We specify a maximum number of cores for the rest of the jobs.
indexer: JIDSlurmOperator = JIDSlurmOperator(
    max_cores=120, task_id="CrystFELIndexer", dag=dag
)
# We can alternatively specify this task be only ever run with the following args.
# indexer: JIDSlurmOperator = JIDSlurmOperator(
#     custom_slurm_params="--partition=milano --ntasks=120 --account=lcls:myaccount",
#     task_id="CrystFELIndexer",
#     dag=dag,
# )

# Merge
merger: JIDSlurmOperator = JIDSlurmOperator(
    max_cores=120, task_id="PartialatorMerger", dag=dag
)

# Figures of merit
hkl_comparer: JIDSlurmOperator = JIDSlurmOperator(
    max_cores=8, task_id="HKLComparer", dag=dag
)
```

Finally, the dependencies between the `Operator`s are "drawn", defining the execution order of the various steps. The `>>` operator has been overloaded for the `Operator` class, allowing it to be used to specify the next step in the DAG. In this case, a completely linear DAG is drawn as:

```py
peak_finder >> indexer >> merger >> hkl_comparer
```

Parallel execution can be added by using the `>>` operator multiple times. Consider a `task1` which upon successful completion starts a `task2` and `task3` in parallel. This dependency can be added to the DAG using:

```py
#task1: JIDSlurmOperator = JIDSlurmOperator(...)
#task2 ...

task1 >> task2
task1 >> task3
```

As each DAG is defined in pure Python, standard control structures (loops, if statements, etc.) can be used to create more complex workflow arrangements.

**Note:** Your DAG will not be available to Airflow until your PR including the file you have defined is merged! Once merged the file will be synced with the Airflow instance and can be run using the scripts described earlier in this document. For testing it is generally preferred that you run each step of your DAG individually using the `submit_slurm.sh` script and the independent **managed** `Task` names. If, however, you want to test the behaviour of Airflow itself (in a modified form) you can use the advanced run-time DAGs defined below as well.

# Advanced Usage
## Run-time DAG creation
In most cases, standard DAGs should be defined as described above and called by name. However, Airflow also supports the creation of DAGs dynamically, e.g. to vary the input data to various steps, or the number of steps that will occur. Some of this functionality has been used to allow for user-defined DAGs which are passed in the form of a dictionary, allowing Airflow to construct the workflow as it is running.

A basic YAML syntax is used to construct a series of nested dictionaries which define a DAG. Considering the first example DAG defined above (for serial femtosecond crystallography), the standard DAG looked like:

```python
peak_finder >> indexer >> merger >> hkl_comparer
```

We can alternatively define this DAG in YAML:

```yaml
task_name: PeakFinderPyAlgos
slurm_params: ''
next:
- task_name: CrystFELIndexer
  slurm_params: ''
  next: []
  - task_name: PartialatorMerger
    slurm_params: ''
    next: []
    - task_name: HKLComparer
      slurm_params: ''
      next: []
```

I.e. we define a tree where each node is constructed using `Node(task_name: str, slurm_params: str, next: List[Node])`. 

- The `task_name` is the name of a **managed** `Task` as before, in the same way that would be passed to the `JIDSlurmOperator`.
- A custom string of slurm arguments can be passed using `slurm_params`. This is a complete string of **all** the arguments to use for the corresponding **managed** `Task`. Use of this field is **all or nothing!** - if it is left as an empty string, the default parameters (passed on the command-line using the launch script) are used, otherwise this string is used in its stead. Because of this **remember to include a partition and account** if using it.
- The `next` field is composed of either an empty list (meaning no **managed** `Task`s are run after the current node), or additional nodes. All nodes in the list are run in parallel. 

As a second example, to run `task1` followed by `task2` and `task3` in parellel we would use:

```yaml
task_name: Task1
slurm_params: ''
next:
- task_name: Task2
  slurm_params: ''
  next: []
- task_name: Task3
  slurm_params: ''
  next: []
```

In order to run a DAG defined this way we pass the **path** to the YAML file we have defined it in to the launch script using `-W <path_to_dag>`. This is instead of calling it by name. E.g.

```bash
/path/to/lute/launch_scripts/submit_launch_airflow.sh /path/to/lute/launch_scripts/launch_airflow.py -e <exp> -r <run> -c /path/to/config -W <path_to_dag> --test [--debug] [SLURM_ARGS]
```

Note that fewer options are currently supported for configuring the operators for each step of the DAG. The slurm arguments can be replaced in their entirety using a custom `slurm_params` string but individual options cannot be modified.
