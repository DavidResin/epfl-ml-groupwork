sshpass -p "dentDoche92i" scp -r project2/Keras admin@192.168.0.21:~/project2
sshpass -p "dentDoche92i" ssh -t admin@192.168.0.21 'cd project2/Keras;python3.6 test.py'