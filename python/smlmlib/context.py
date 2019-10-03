# -*- coding: utf-8 -*-

import ctypes
from .base import SMLM

class Context:
    def __init__(self, smlm:SMLM):
        self.smlm = smlm
        lib = smlm.lib
        self.lib = lib
        
        self._Context_Create = lib.Context_Create
        self._Context_Create.argtypes=[]
        self._Context_Create.restype = ctypes.c_void_p
        
        self._Context_Destroy = lib.Context_Destroy
        self._Context_Destroy.argtypes = [ctypes.c_void_p]
        
        self.inst = self._Context_Create()

    def Destroy(self):
        if self.inst:
            self._Context_Destroy(self.inst)
            self.inst=None
            
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.Destroy()


        