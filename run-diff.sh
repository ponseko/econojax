COMMON_ARGS="-t 10000000 -a 100 -e 10 -w --trade_prices 3 6 9 --craft_diff_resources_required 0 --trade_expiry_time 20 --rollout_length 150 --eval_runs 8 --network_size_pop_policy 256 256 --network_size_pop_value 256 256"
WANDB_GROUP_NAME="new--ctde-100"
for seed in {1..3} 
do for num_resources in {2..20..2} 
do
    python train.py $COMMON_ARGS -s $seed -ps $seed -wg $WANDB_GROUP_NAME-r-$num_resources -r $num_resources;
    python train.py $COMMON_ARGS -s $seed -ps $seed -wg $WANDB_GROUP_NAME-r-$num_resources -r $num_resources -i --network_size_pop_policy 128 128; # CTDE
    python train.py $COMMON_ARGS -s $seed -ps $seed -wg $WANDB_GROUP_NAME-r-$num_resources -r $num_resources -i -iv --network_size_pop_policy 128 128 --network_size_pop_value 256 256;
    python train.py $COMMON_ARGS -s $seed -ps $seed -wg $WANDB_GROUP_NAME-r-$num_resources -r $num_resources --insert_agent_ids True;
    python train.py $COMMON_ARGS -s $seed -ps $seed -wg $WANDB_GROUP_NAME-r-$num_resources -r $num_resources --insert_agent_ids True -i --network_size_pop_policy 128 128; # CTDE
done
done
