#!/bin/bash

filename=startp
# filename=astronaut

read x y z <<< $(echo `python image_save_as_text.py $filename`)
echo "image converted to text file"

mpicc image_blur_parallel.c -o image_blur_parallel
echo "image_blur.c compiled"

mpirun -np 4 image_blur_parallel $filename $x $y $z
echo "image_blur executed"

python image_load_from_txt.py $filename $x $y $z
echo "blurred image plotted"
