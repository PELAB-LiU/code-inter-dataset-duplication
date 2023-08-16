
# iterate over java-small, java-med
for DATASET in java-med ; do
    # download the dataset
    wget https://s3.amazonaws.com/code2seq/datasets/${DATASET}.tar.gz
    # extract the dataset
    tar -xvf ${DATASET}.tar.gz
    # remove the archive
    rm ${DATASET}.tar.gz
done