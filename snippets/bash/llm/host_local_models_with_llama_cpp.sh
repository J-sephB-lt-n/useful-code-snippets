<<'###BLOCK-COMMENT'
TAGS: chat|gpt|host|hosting|language|large language model|llama|llm|local|nlp 
DESCRIPTION: Hosting a Large Language Model (LLM) locally using llama.cpp
###BLOCK-COMMENT

# build llama.cpp
mkd ~/llama_cpp
cd ~/llama_cpp
git clone https://github.com/ggerganov/llama.cpp
cd ~/llama_cpp/llama.cpp
make

# launch model server
PATH_TO_MODEL='/Users/josephbolton/local_llms/bartowski/dolphin-2.9.3-mistral-7B-32k-GGUF/dolphin-2.9.3-mistral-7B-32k-IQ4_XS.gguf'
./llama-server -m $PATH_TO_MODEL --port 8080 --ctx-size 999
# now, can access GUI at http://localhost:8080
# ...or can use chat completion endpoint:
curl -X POST http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{
  "prompt": "Once upon a time",
  "n_predict": 50   
}'
# Llama-server documentation here: https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md
