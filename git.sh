#!/bin/bash

# $Id$

git add *.[FChm] *.pl *.sh Makefile *.cu

#svnId *.[FChm] *.pl *.sh *.inc Makefile

exit

git commit -m "comments"

git push origin master

