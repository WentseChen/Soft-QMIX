#!/bin/bash
debug=
# debug=echo
trap 'onCtrlC' INT

function onCtrlC () {
  echo 'Ctrl+C is captured'
  for pid in $(jobs -p); do
    kill -9 $pid
  done

  kill -HUP $( ps -A -ostat,ppid | grep -e '^[Zz]' | awk '{print $2}')
  exit 1
}

config=$1  # qmix
tag=$2
maps=${3:-sc2_gen_protoss,sc2_gen_terran,sc2_gen_zerg}   # MMM2 left out
units=${8:-10,5,20} # include 5 and 20 units if possible
threads=${4:-1} # 2
td_lambdas=${9:-0.4}
eps_anneals=${10:-100000}
args=${5:-}    # ""
gpus=${11:-0,1,2,3,4,5,6,7}
times=${7:-3}  # could change to 1 and only run 1 seed for each unit type as well

maps=(${maps//,/ })
units=(${units//,/ })
args=(${args//,/ })
td_lambdas=(${td_lambdas//,/ })
eps_anneals=(${eps_anneals//,/ })
gpus=(${gpus//,/ })

if [ ! $config ] || [ ! $tag ]; then
    echo "Please enter the correct command."
    echo "bash run.sh config_name map_name_list (experinments_threads_num arg_list gpu_list experinments_num)"
    exit 1
fi

# run parallel
count=0
for tdlambda in "${td_lambdas[@]}"; do
    for epsanneal in "${eps_anneals[@]}"; do
        for map in "${maps[@]}"; do
            for unit in "${units[@]}"; do
                for((i=0;i<times;i++)); do
                    group="${config}-${map}-${tag}"
                    gpu=${gpus[$(($count % ${#gpus[@]}))]}
                    $debug ./run_docker.sh $gpu python3 src/main.py --config="$config" --env-config="$map" with group="$group" env_args.capability_config.n_units=$unit env_args.capability_config.n_enemies=$unit use_wandb=True td_lambda=$tdlambda epsilon_anneal_time=$epsanneal save_model=True "${args[@]}" &

                    count=$(($count + 1))
                    if [ $(($count % $threads)) -eq 0 ]; then
                        wait
                    fi
                    # for random seeds
                done
            done
        done
    done
done
wait
