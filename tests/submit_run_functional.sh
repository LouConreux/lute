#!/bin/bash

#!/bin/bash
usage()
{
    cat << EOF
run_functional.py [-h] [-a] [--git_pr_id GIT_PR_ID] [--git_tag GIT_TAG] [--no_delete] -r RUN_DIR [--run_tests RUN_TESTS] [--tests_dir TESTS_DIR] [--test_airflow] [--use_local_tests] [--use_prefect]
    Run a series of functional tests for LUTE.

    Options:
        # Airflow and prefect arguments
        ###############################

        -a|--admin
          Use an administrator account for Airflow authentication. Ignored if using prefect. Default: False

        --test_airflow
          Use the test Airflow instance. Ignored if using prefect. Default: False

        --use_prefect
          Use prefect instead of Airflow.

        # Options to select a version of LUTE to use for the tests and where to install.
        ################################################################################

        --git_pr_id GIT_PR_ID
          Checkout a PR branch to run LUTE against based on GitHub ID. (Optional)

        --git_tag GIT_TAG
          Checkout a specific tag (e.g. release) of LUTE. (Optional)

        -r|--run_dir RUN_DIR
          Directory to install LUTE in, and setup the output folder. (Required)

        # Options to select specific  tests and configure behaviour of tests.
        #####################################################################

        --no_delete
          If passed, do not delete output files when tests are finished.

        --run_tests RUN_TESTS
          Provide a comma-separated string of tests to run. If provided, this script
          will only run those, rather than the default behaviour of running all tests.
          E.g: --run_these_tests test2,test5. Tests that do not exist are silently ignored.

        --tests_dir TESTS_DIR
          Specify an alternative path to tests than those from the LUTE clone. Must have the same directory structure:
          $DIR/test1/... $DIR/test2/... If this flag and --use_local_tests are both passed, this one is used.

        --use_local_tests
          Use the tests from the installation of LUTE where this script is called, rather than those from the clone of LUTE which is run against,
          or another directory if passed. If this flag and --tests_dir are both passed, --tests_dir is used.

        # Misc.
        #######

        -h|--help
          Display this message.
EOF
}

SLURM_ACCOUNT="lcls:data"

while [[ $# -gt 0 ]]
do
    flag="$1"

    case $flag in
        --account=*)
            SLURM_ACCOUNT="${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            exit
            ;;
        --run_dir)
            POS+=("$1")
            POS+=("$2")
            RUN_DIR="$2"
            shift
            shift
            ;;
        *)
            POS+=("$1")
            shift
            ;;
    esac
done
set -- "${POS[@]}"

if [[ -z ${RUN_DIR} ]]; then
    echo "Must provide a run directory!"
    usage
    exit
fi

# Bodge Kerberos credentials
# These duplicates are removed later by the workflow process
KERB_CACHE_PATH=$(klist -l | awk -F"FILE:" '{printf (NF>1)? $NF : ""}')
if [[ ! -d $HOME/.tmp_cache ]]; then
    mkdir $HOME/.tmp_cache
fi
cp $KERB_CACHE_PATH $HOME/.tmp_cache/kerbcache
export KRB5CCNAME="FILE:${HOME}/.tmp_cache/kerbcache"

TEST_SCRIPT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh
echo "Sourced latest psconda.sh"
CMD="python -B ${TEST_SCRIPT_PATH}/run_functional.py ${@}"
echo $CMD
SLURM_ARGS="--account=${SLURM_ACCOUNT} --ntasks=1 --partition=milano"
echo "Running ${CMD} with ${SLURM_ARGS}"
sbatch $SLURM_ARGS --wrap "${CMD}"
export KRB5CCNAME="FILE:${KERB_CACHE_PATH}"
