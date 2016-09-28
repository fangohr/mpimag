#!/bin/bash

# filename=start
filename=astronaut

read x y z <<< $(echo `python image_save_as_text.py $filename`)
echo "image converted to text file"

gcc image_blur.c -o image_blur
echo "image_blur.c compiled"

./image_blur $filename $x $y $z
echo "image_blur executed"

python image_load_from_txt.py $filename $x $y $z
echo "blurred image plotted"
