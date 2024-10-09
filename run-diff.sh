COMMON_ARGS="-t 10000000 -a 100 -e 4 -w --trade_prices 3 6 9 --craft_diff_resources_required 0 --trade_expiry_time 10 --rollout_length 300 --eval_runs 10"
WANDB_GROUP_NAME="scaled"
for seed in {1..3} 
do for num_resources in {2..16..2} 
do
    python train.py $COMMON_ARGS -s $seed -ps $seed -wg $WANDB_GROUP_NAME-r-$num_resources -r $num_resources;
    python train.py $COMMON_ARGS -s $seed -ps $seed -wg $WANDB_GROUP_NAME-r-$num_resources -r $num_resources -i;
    python train.py $COMMON_ARGS -s $seed -ps $seed -wg $WANDB_GROUP_NAME-r-$num_resources -r $num_resources --insert_agent_ids True;
done
done