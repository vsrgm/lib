#!/bin/sh

a=11

while [ $a -gt 10 ]
do
   echo $a
   a=`expr $a + 1`
   ./v4l2_camera
done
