for i in {1..10}
do
    nohup python train.py > train_output_$i.log 2>&1
done