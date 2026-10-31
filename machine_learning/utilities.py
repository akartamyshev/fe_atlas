#!/usr/bin/python3
# -*- coding: utf-8 -*- 

def makedir(path, renew_folder=False):
    """
    *path* - path to some file 
    Make dirname(path) directory if it does not exist
    """
    import os
    import shutil as S

    dirname = os.path.dirname(path)

    if renew_folder and os.path.exists(dirname): S.rmtree(path)
    else: pass

    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)
        print("Directory", dirname, " was prepared")
    return