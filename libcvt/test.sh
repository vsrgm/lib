count=1
while [ $count -lt 360 ]
do
  ./a.out ./Testapp/Horizontal_bar.yuv 640 480 $count sample.yuv
  count=`expr $count + 1`
  sleep 1
done
