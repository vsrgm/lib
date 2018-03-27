echo -n { >face.txt
grep  "f " data/elepham.obj | cut -d ' ' -f 2,3,4 | sed 's/\//,/g' | sed 's/\ /,/g' |sed 's/,,/,0,/g' | sed ':a;N;$!ba;s/\n/},\n{/g'>> face.txt
truncate -s-2 face.txt
printf } >>face.txt
dos2unix face.txt

echo -n { > vertex.txt
grep  "v " data/elepham.obj | cut -d ' ' -f 3,4,5 |sed 's/ /,/g' |sed ':a;N;$!ba;s/\n/},\n{/g' >> vertex.txt
truncate -s-2 vertex.txt
printf } >>vertex.txt
dos2unix vertex.txt

echo -n { > vertexnormal.txt
grep  "vn " data/elepham.obj | cut -d ' ' -f 3,4,5 |sed 's/ /,/g' |sed ':a;N;$!ba;s/\n/},\n{/g' >> vertexnormal.txt
truncate -s-2 vertexnormal.txt
printf } >>vertexnormal.txt
dos2unix vertexnormal.txt
