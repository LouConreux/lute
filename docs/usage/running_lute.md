# Running `LUTE`
## Running Managed `Task`s and Workflows (DAGs)
After a YAML file has been filled in you can run a `Task` (really, a **managed** `Task`). There are multiple ways to submit a `Task`, but there are 3 that are most likely:

1. Run a single **managed** `Task` interactively by running `python ...`
2. Run a single **managed** `Task` as a batch job (e.g. on S3DF) via a SLURM submission `submit_slurm.sh ...`
3. Run a DAG (workflow with multiple **managed** `Task`s).

These will be covered in turn below; however, in general all methods will require two parameters: the path to a configuration YAML file, and the name of the **managed** `Task` or workflow you want to run. When submitting via SLURM or submitting an entire workflow there are additional parameters to control these processes.


### Running single managed `Task`s interactively
The simplest submission method is just to run Python interactively. In most cases this is not practical for long-running analysis, but may be of use for short `Task`s or when debugging. From the root directory of the LUTE repository (or after installation) you can use the `run_task.py` script:

```bash
> python -B [-O] run_task.py -t <ManagedTaskName> -c </path/to/config/yaml>
```

The command-line arguments in square brackets `[]` are optional, while those in `<>` must be provided:

- `-O` is the flag controlling whether you run in debug or non-debug mode. **By default, i.e. if you do NOT provide this flag you will run in debug mode** which enables verbose printing. Passing `-O` will turn off debug to minimize output.
- `-t <ManagedTaskName>` is the name of the **managed** `Task` you want to run.
- `-c </path/...>` is the path to the configuration YAML.

#### Gotchas!

Be sure to modify your configuration file to include an `experiment` and `run` in the header section! Many `Task`s (including some code outside of LUTE's control) will require these values to work properly. Alternatively you can provide them as environment variables when running your command, e.g.:

```bash
> EXPERIMENT="..." RUN_NUM=123 python -B [-O] -t <ManagedTaskName> -c <path/to/config.yaml>
```

If you are submitting a **managed** `Task` that uses MPI you may encounter an issue at launch due to rank selection. Depending on where you are running this could be because the fallback mechanisms for determining the number of cores are inconsistent with the true number of cores available to your processes. In order to get around this you can "mimic" the SLURM submission process by setting the environment variable `SLURM_NPROCS` to some small number of cores (e.g. 2)

```bash
> SLURM_NPROCS=2 python -B [-O] -t <ManagedTaskName> -c <path/to/config.yaml>
```

For additional debugging variables see the advanced usage section below.

### Submitting a single managed `Task` as a batch job
On S3DF you can also submit individual **managed** `Task`s to run as batch jobs. To do so use `launch_scripts/submit_slurm.sh`

```bash
> launch_scripts/submit_slurm.sh -t <ManagedTaskName> -c </path/to/config/yaml> [--debug] $SLURM_ARGS
```

As before command-line arguments in square brackets `[]` are optional, while those in `<>` must be provided

- `-t <ManagedTaskName>` is the name of the **managed** `Task` you want to run.
- `-c </path/...>` is the path to the configuration YAML.
- `--debug` is the flag to control whether or not to run in debug mode.

In addition to the LUTE-specific arguments, SLURM arguments must also be provided (`$SLURM_ARGS` above). You can provide as many as you want; however you will need to at least provide:

- `--partition=<partition/queue>` - The queue to run on, in general for LCLS this is `milano`
- `--account=lcls:<experiment>` - The account to use for batch job accounting.

You will likely also want to provide at a minimum:

- `--ntasks=<...>` to control the number of cores in allocated.

In general, it is best to prefer the long-form of the SLURM-argument (`--arg=<...>`) in order to avoid potential clashes with present or future LUTE arguments.

#### Experiment and run, again...

If you are not providing a specific experiment and run in your configuration YAML, you can additionally pass these values as arguments on the command-line:

```bash
> launch_scripts/submit_slurm.sh -t <ManagedTaskName> -c </path/to/config/yaml> [-e EXPERIMENT] [-r RUN] [--debug] $SLURM_ARGS
```

### Workflow (DAG) submission
Finally, you can submit a full workflow (e.g. SFX analysis, smalldata production and summary results, geometry optimization...). This can be done using a single script, `submit_launch_airflow.sh`, similarly to the SLURM submission above:

```bash
> launch_scripts/submit_launch_airflow.sh /path/to/lute/launch_scripts/launch_airflow.py -c </path/to/yaml.yaml> -w <dag_name> [--debug] [--test] [-e <exp>] [-r <run>] $SLURM_ARGS
```
The submission process is slightly more complicated in this case. A more in-depth explanation is provided under "Airflow Launch Steps", in the advanced usage section below if interested. The parameters are as follows - as before command-line arguments in square brackets `[]` are optional, while those in `<>` must be provided:

- The **first argument** (must be first) is the full path to the `launch_scripts/launch_airflow.py` script located in whatever LUTE installation you are running. All other arguments can come afterwards in any order.
- `-c </path/...>` is the path to the configuration YAML to use.
- `-w <dag_name>` is the name of the DAG (workflow) to run. This replaces the task name provided when using the other two methods above. A DAG list is provided below.
  - **NOTE:** For advanced usage, a custom DAG can be provided at **run** time using `-W` (capital W) followed by the path to the workflow instead of `-w`. See below for further discussion on this use case.
- `--debug` controls whether to use debug mode (verbose printing)
- `--test` controls whether to use the test or production instance of Airflow to manage the DAG. The instances are running identical versions of Airflow, but the `test` instance may have "test" or more bleeding edge development DAGs.
- `-e` is used to pass the experiment name. Needed if not using the ARP, i.e. running from the command-line.
- `-r` is used to pass a run number. Needed if not using the ARP, i.e. running from the command-line.

The `$SLURM_ARGS` must be provided in the same manner as when submitting an individual **managed** `Task` by hand to be run as batch job with the script above. **Note** that these parameters will be used as the starting point for the SLURM arguments of **every managed** `Task` in the DAG; however, individual steps in the DAG may have overrides built-in where appropriate to make sure that step is not submitted with potentially incompatible arguments. For example, a single threaded analysis `Task` may be capped to running on one core, even if in general everything should be running on 100 cores, per the SLURM argument provided. These caps are added during development and cannot be disabled through configuration changes in the YAML.

**Note for LCLS Staff**: LCLS staff should refer to the Advanced Usaged section for information on accessing Airflow with greater privileges.

**DAG List**

- `find_peaks_index`
- `psocake_sfx_phasing`
- `pyalgos_sfx`
- `smd_summaries`

#### DAG Submission from the `eLog`
You can use the script in the previous section to submit jobs through the eLog. To do so navigate to the `Workflow > Definitions` tab using the blue navigation bar at the top of the eLog. On this tab, in the top-right corner (underneath the help and zoom icons) you can click the `+` sign to add a new workflow. This will bring up a "Workflow definition" UI window. When filling out the eLog workflow definition the following fields are needed (all of them):

- `Name`: You can name the workflow anything you like. It should probably be something descriptive, e.g. if you are using LUTE to run smalldata_tools, you may call the workflow `lute_smd`.
- `Executable`: In this field you will put the **full path** to the `submit_launch_airflow.sh` script:  `/path/to/lute/launch_scripts/submit_launch_airflow.sh`.
- `Parameters`: You will use the parameters as described above. Remember the first argument will be the **full path** to the `launch_airflow.py` script (this is NOT the same as the bash script used in the executable!): `/full/path/to/lute/launch_scripts/launch_airflow.py -c <path/to/yaml> -w <dag_name> [--debug] [--test] $SLURM_ARGS`
- `Location`: **Be sure to set to** `S3DF`.
- `Trigger`: You can have the workflow trigger automatically or manually. Which option to choose will depend on the type of workflow you are running. In general the options `Manually triggered` (which displays as `MANUAL` on the definitions page) and `End of a run` (which displays as `END_OF_RUN` on the definitions page) are safe options for ALL workflows. The latter will be automatically submitted for you when data acquisition has finished. If you are running a workflow with **managed** `Task`s that work as data is being acquired (e.g. `SmallDataProducer`), you may also select `Start of a run` (which displays as `START_OF_RUN` on the definitions page).

Upon clicking create you will see a new entry in the table on the definitions page. In order to run `MANUAL` workflows, or re-run automatic workflows, you must navigate to the `Workflows > Control` tab. For each acquisition run you will find a drop down menu under the `Job` column. To submit a workflow you select it from this drop down menu by the `Name` you provided when creating its definition.

## Advanced Usage
### Airflow Launch and DAG Execution Steps
The Airflow launch process actually involves two steps. There is a wrapper prior to getting to the actual Airflow API communication.

1. `launch_scripts/submit_launch_airflow.sh` is run.
2. This script runs the `launch_scripts/launch_airflow.py` script which was provided as the first argument. This is the **true** launch script
3. `launch_airflow.py` communicates with the Airflow API, requesting that a specific DAG be launched. It then continues to run, and gathers the individual logs and the exit status of each step of the DAG.
4. Airflow will then enter a loop of communication where it asks the JID to submit each step of the requested DAG as batch job using `launch_scripts/submit_slurm.sh`.

There are some specific reasons for this complexity:

- The use of `submit_launch_airflow.sh` is to allow the true Airflow launch script to be a long-lived job. This is for compatibility with the eLog and the ARP. When run from the eLog as a workflow, the job submission process must occur within 30 seconds due to a timeout built-in to the system. This is fine when submitting jobs to run on the batch-nodes, as the submission to the queue takes very little time. So here, `submit_launch_airflow.sh` serves as a thin script to have `launch_airflow.py` run as a batch job. It can then run as a long-lived job (for the duration of the entire DAG) collecting log files all in one place. This allows the log for each stage of the Airflow DAG to be inspected in a single file, and through the eLog browser interface.


#### Elevated Privileges
The `launch_airflow.py` script (and by proxy the `submit_launch_airflow.sh` script) can be run as a user with greater privileges. This involves passing an additional flag `--admin` to the script. You need sufficient permissions to access the credentials to use this account which currently means membership of the `ps-data` Unix group.

### Custom Run-Time DAGs
In most cases, standard DAGs should be called as described above. However, Airflow also supports the dynamic creation of DAGs, e.g. to vary the input data to various steps, or the number of steps that will occur. Some of this functionality has been used to allow for user-defined DAGs which are passed in the form of a dictionary, allowing Airflow to construct the workflow as it is running.

A basic YAML syntax is used to construct a series of nested dictionaries which define a DAG. Consider a simplified serial femtosecond crystallography DAG which runs peak finding through merging and then calculates some statistics. I.e. we want an execution order that looks like:

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
      next:
```

I.e. we define a tree where each node is constructed using `Node(task_name: str, slurm_params: str, next: List[Node])`.

- The `task_name` is the name of a **managed** `Task`. This name **must** be identical to a **managed** `Task` defined in the LUTE installation you are using.
- A custom string of slurm arguments can be passed using `slurm_params`. This is a complete string of **all** the arguments to use for the corresponding **managed** `Task`. Use of this field is **all or nothing!** - if it is left as an empty string, the default parameters (passed on the command-line using the launch script) are used, otherwise this string is used in its stead. Because of this **remember to include a partition and account** if using it.
- The `next` field is composed of either an empty list (meaning no **managed** `Task`s are run after the current node), or additional nodes. All nodes in the `next` list are run in parallel.

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

In order to run a DAG defined in this way, we pass the **path** to the YAML file we have defined it in to the launch script using `-W <path_to_dag>`. This is instead of calling it by name. E.g.

```bash
/path/to/lute/launch_scripts/submit_launch_airflow.sh /path/to/lute/launch_scripts/launch_airflow.py -e <exp> -r <run> -c /path/to/config -W <path_to_dag> --test [--debug] [SLURM_ARGS]
```

Note that fewer options are currently supported for configuring the operators for each step of the DAG.  The slurm arguments can be replaced in their entirety using a custom `slurm_params` string but individual options cannot be modified.

### Debug Environment Variables
Special markers have been inserted at certain points in the execution flow for LUTE. These can be enabled by setting the environment variables detailed below. These are intended to allow developers to exit the program at certain points to investigate behaviour or a bug. For instance, when working on configuration parsing, an environment variable can be set which exits the program after passing this step. This allows you to run LUTE otherwise as normal (described above), without having to modify any additional code or insert your own early exits.

Types of debug markers:

- `LUTE_DEBUG_EXIT`: Will exit the program at this point if the corresponding environment variable has been set.

Developers can insert these markers as needed into their code to add new exit points, although as a rule of thumb they should be used sparingly, and generally only after major steps in the execution flow (e.g. after parsing, after beginning a task, after returning a result, etc.).

In order to include a new marker in your code:
```py
from lute.execution.debug_utils import LUTE_DEBUG_EXIT

def my_code() -> None:
    # ...
    LUTE_DEBUG_EXIT("MYENVVAR", "Additional message to print")
    # If MYENVVAR is not set, the above function does nothing
```

You can enable a marker by setting to 1, e.g. to enable the example marker above while running `Tester`:
```bash
MYENVVAR=1 python -B run_task.py -t Tester -c config/test.yaml
```

#### Currently used environment variables
- `LUTE_DEBUG_EXIT_AT_YAML`: Exits the program after reading in a YAML configuration file and performing variable substitutions, but BEFORE Pydantic validation.
- `LUTE_DEBUG_BEFORE_TPP_EXEC`: Exits the program after a ThirdPartyTask has prepared its submission command, but before `exec` is used to run it.
