
# iterate over java-small, java-med, java-large
for DATASET in java-small java-med java-large ; do
    # download the dataset
    wget https://s3.amazonaws.com/code2seq/datasets/${DATASET}.tar.gz
    # extract the dataset
    tar -xvf ${DATASET}.tar.gz
    # remove the archive
    rm ${DATASET}.tar.gz
done