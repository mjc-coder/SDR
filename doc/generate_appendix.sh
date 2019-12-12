#!/bin/bash


## Generates the appendix for the wsu thesis latex.  Essentially a refman.tex but stripped down.

#Generate Prefix
echo "%% Autogenerated File Do Not Modify %%" > $2
echo "%--- Begin generated contents ---" >> $2
echo "\graphicspath{{../../$1/}}" >> $2

##
if [ -e $1/hierarchy.tex ]; then
  echo "\section{Hierarchical Index}" >> $2
  echo "\input{../../$1/hierarchy.tex}" >> $2
fi

##
if [ -e $1/annotated.tex ]; then
  echo "\section{Class Index}" >> $2
  echo "\input{../../$1/annotated.tex}" >> $2
fi

##
echo "\section{Class Documentation}" >> $2
for f in $(ls $1/*.tex | grep -vi hierarchy | grep -vi annotated | grep -vi refman)
do
  echo "\input{../../$f}" >> $2
  # do something on $f
done
echo "%--- End of generated contents ---" >> $2

