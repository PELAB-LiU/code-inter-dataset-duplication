
wget https://raw.githubusercontent.com/microsoft/CodeXGLUE/main/Code-Code/code-to-code-trans/data/test.java-cs.txt.java
wget https://raw.githubusercontent.com/microsoft/CodeXGLUE/main/Code-Code/code-to-code-trans/data/valid.java-cs.txt.java
wget https://raw.githubusercontent.com/microsoft/CodeXGLUE/main/Code-Code/code-to-code-trans/data/train.java-cs.txt.java
wget https://raw.githubusercontent.com/microsoft/CodeXGLUE/main/Code-Code/code-to-code-trans/data/test.java-cs.txt.cs
wget https://raw.githubusercontent.com/microsoft/CodeXGLUE/main/Code-Code/code-to-code-trans/data/valid.java-cs.txt.cs
wget https://raw.githubusercontent.com/microsoft/CodeXGLUE/main/Code-Code/code-to-code-trans/data/train.java-cs.txt.cs

cat train.java-cs.txt.java valid.java-cs.txt.java test.java-cs.txt.java > all.java
cat train.java-cs.txt.cs valid.java-cs.txt.cs test.java-cs.txt.cs > all.cs

rm *.txt.java
rm *.txt.cs
