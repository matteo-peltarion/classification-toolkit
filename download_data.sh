#!/bin/bash

#wget https://storage.googleapis.com/peltarion-ml-assignment/HAM10000/HAM10000_images_part_1.zip
#wget https://storage.googleapis.com/peltarion-ml-assignment/HAM10000/HAM10000_images_part_2.zip
#wget https://storage.googleapis.com/peltarion-ml-assignment/HAM10000/HAM10000_metadata.csv
unzip -qq HAM10000_images_part_1.zip -d data
unzip -qq HAM10000_images_part_2.zip -d data
mv HAM10000_metadata.csv data/
