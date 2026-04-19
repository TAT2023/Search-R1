
file_path=/root/shared-nvme/wiki
index_file=$file_path/bm25
corpus_file=$file_path/wiki-18.jsonl
retriever_name=bm25
LOG_FILE=retrieval_server.log

# BM25 retrieval runs on CPU and does not require CUDA_VISIBLE_DEVICES.

nohup python search_r1/search/retrieval_server.py --index_path $index_file \
                                                   --corpus_path $corpus_file \
                                                   --topk 3 \
                                                   --retriever_name $retriever_name \
                                                   > $LOG_FILE 2>&1 &

RETRIEVAL_PID=$!
echo "Retrieval server started in background. PID=$RETRIEVAL_PID, log=$LOG_FILE"
