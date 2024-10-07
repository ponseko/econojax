for seed in {2..2} 
do for num_resources in {6..10..2} 
do
    python train.py -s $seed -ps $seed -t 10000000 -a 40 -e 6 -w -wg resource-$num_resources -r $num_resources --trade_prices 3 6 9 --craft_diff_resources_required $(( (num_resources / 2)));
    python train.py -s $seed -ps $seed -t 10000000 -a 40 -e 6 -w -wg resource-$num_resources -r $num_resources -i --trade_prices 3 6 9 --craft_diff_resources_required $(( (num_resources / 2)));
    python train.py -s $seed -ps $seed -t 10000000 -a 40 -e 6 -w -wg resource-$num_resources -r $num_resources --insert_agent_ids True --trade_prices 3 6 9 --craft_diff_resources_required $(( (num_resources / 2)));
done
done