textattack attack --disable-stdout --model bert-base-uncased-imdb --model-batch-size 32 --num-workers-per-device 1 --recipe a2t --num-examples 1000 --log-to-csv runs/baselines/a2t-sent0.9-imdb1000.csv
textattack attack --disable-stdout --model bert-base-uncased-yelp --model-batch-size 32 --num-workers-per-device 1 --recipe a2t --num-examples 1000 --log-to-csv runs/baselines/a2t-sent0.9-yelp1000.csv
textattack attack --disable-stdout --model bert-base-uncased-mnli --model-batch-size 32 --num-workers-per-device 1 --recipe a2t --num-examples -1 --log-to-csv runs/baselines/a2t-sent0.9-mnli.csv
