
gdown 'https://drive.google.com/uc?id=1VKngMe1WByGtevTGV1-IUyKqi12RkzKs' -O 'python-150.rar'
unrar x 'python-150.rar'
rm 'python-150.rar'
cat code\ search/NCS/doc_code_dataset/code_train.txt code\ search/NCS/doc_code_dataset/code_test.txt code\ search/NCS/doc_code_dataset/code_valid.txt > snippets.txt
cat code\ search/NCS/doc_code_dataset/comment_train.txt code\ search/NCS/doc_code_dataset/comment_test.txt code\ search/NCS/doc_code_dataset/comment_valid.txt > descriptions.txt
rm -r code\ search/