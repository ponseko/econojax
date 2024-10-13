COMMON_ARGS="-t 8000000 -a 100 -e 10 -w --trade_prices 3 6 9 --craft_diff_resources_required 0 --trade_expiry_time 20 --rollout_length 150 --eval_runs 8 --network_size_pop 256 256"
WANDB_GROUP_NAME="new-100"
for seed in {1..3} 
do for num_resources in {2..20..2} 
do
    python train.py $COMMON_ARGS -s $seed -ps $seed -wg $WANDB_GROUP_NAME-r-$num_resources -r $num_resources;
    python train.py $COMMON_ARGS -s $seed -ps $seed -wg $WANDB_GROUP_NAME-r-$num_resources -r $num_resources -i --network_size_pop 128 128;
    python train.py $COMMON_ARGS -s $seed -ps $seed -wg $WANDB_GROUP_NAME-r-$num_resources -r $num_resources --insert_agent_ids True;
done
done
