textattack attack --disable-stdout --model bert-base-uncased-imdb ---model-batch-size 64 --num-workers-per-device 2 --parallel -recipe bert-attack --num-examples -1 --log-to-csv runs/baselines/bert-attack.csv
textattack attack --disable-stdout --model bert-base-uncased-imdb ---model-batch-size 64 --num-workers-per-device 2 --parallel -recipe textfooler --num-examples -1 --log-to-csv runs/baselines/textfooler.csv
textattack attack --disable-stdout --model bert-base-uncased-imdb --model-batch-size 64 --num-workers-per-device 2 --parallel --recipe a2t --num-examples -1 --log-to-csv runs/baselines/a2t.csv