set -o allexport
source .env set
+o allexport
# download the data required for executing evaluation
huggingface-cli download --repo-type=dataset --local-dir=${DATA_PATH}/downloads/extracted/ TIGER-Lab/SKGInstruct ./skg_raw_data.zip
# unzip it in that folder
unzip -o ${DATA_PATH}/downloads/extracted/skg_raw_data.zip -d ${DATA_PATH}/downloads/extracted/

# download the test data
# NOTE: the 7b and 13/34b has a slightly different eval format due to a slight training bug that does not affect performance
huggingface-cli download --repo-type=dataset --local-dir=${DATA_PATH}/processed/ TIGER-Lab/SKGInstruct ./skginstruct_test_file_7b.json 