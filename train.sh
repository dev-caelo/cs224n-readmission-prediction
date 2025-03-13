# conda activate cs224n

# echo "running RoBERTa 2"
# python3 modified_train.py "RoBERTa" 2

# echo "running RoBERTa 6"
# python3 modified_train.py "RoBERTa" 6

echo "running Clinical_BERT 2"
python3 modified_train.py "Clinical_BERT" 2

echo "running Clinical_BERT 6"
python3 modified_train.py "Clinical_BERT" 6