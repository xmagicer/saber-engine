"""Explicit HIP module dispatch for isolated drafter experiments.

Arguments come from the shipped manifest. Tensor storage must remain live through
stream completion; all calls use PyTorch's current HIP stream on one device.
"""
import ctypes
from pathlib import Path
import tomllib
import torch


class HipModule:
    """Own one loaded code object; never unload while captured graphs may reference it."""
    def __init__(self,directory,plugin,device=0,library='/opt/rocm/lib/libamdhip64.so.7'):
        torch.cuda.set_device(device)
        self.device=device
        self.lib=ctypes.CDLL(library)
        p=ctypes.c_void_p;u=ctypes.c_uint
        self.lib.hipModuleLoad.argtypes=[ctypes.POINTER(p),ctypes.c_char_p];self.lib.hipModuleLoad.restype=ctypes.c_int
        self.lib.hipModuleGetFunction.argtypes=[ctypes.POINTER(p),p,ctypes.c_char_p];self.lib.hipModuleGetFunction.restype=ctypes.c_int
        self.lib.hipModuleLaunchKernel.argtypes=[p,u,u,u,u,u,u,u,p,ctypes.POINTER(p),ctypes.POINTER(p)];self.lib.hipModuleLaunchKernel.restype=ctypes.c_int
        self.lib.hipGetErrorString.argtypes=[ctypes.c_int];self.lib.hipGetErrorString.restype=ctypes.c_char_p
        directory=Path(directory);self.meta=tomllib.loads((directory/(plugin+'.manifest.toml')).read_text())
        if self.meta['arch']!='gfx1201' or self.meta['plugin']!=plugin:raise ValueError('Code object identity mismatch')
        self.kernels={k['name']:k for k in self.meta['kernel']};self.module=p();self.functions={}
        self.check(self.lib.hipModuleLoad(ctypes.byref(self.module),str(directory/self.meta['hsaco']).encode()),'hipModuleLoad')

    def check(self,status,operation):
        if status:
            reason=self.lib.hipGetErrorString(status)
            raise RuntimeError(f'{operation}: HIP {status}: {reason.decode() if reason else "unknown"}')

    def launch(self,name,grid,block,arguments,shared_bytes=0):
        decl=self.kernels[name]
        explicit=[a for a in decl['arg'] if a['value_kind'] in ['global_buffer','by_value']]
        expected={a['name'] for a in explicit}
        if set(arguments)!=expected:raise ValueError(f'{name} arguments: missing {expected-set(arguments)}, extra {set(arguments)-expected}')
        function=self.functions.get(name)
        if function is None:
            function=ctypes.c_void_p();self.check(self.lib.hipModuleGetFunction(ctypes.byref(function),self.module,name.encode()),'hipModuleGetFunction');self.functions[name]=function
        holders=[]
        for arg in explicit:
            value=arguments[arg['name']]
            if arg['value_kind']=='global_buffer':
                if not isinstance(value,torch.Tensor) or not value.is_cuda or value.device.index!=self.device:raise ValueError('Kernel buffers must be tensors on the module device')
                holders.append(ctypes.c_void_p(value.data_ptr()))
            elif isinstance(value,float) and arg['size']==4:holders.append(ctypes.c_float(value))
            elif isinstance(value,int) and arg['size']==4:holders.append(ctypes.c_int32(value))
            elif isinstance(value,int) and arg['size']==8:holders.append(ctypes.c_int64(value))
            else:raise ValueError('Unsupported scalar ABI: '+str(arg))
        params=(ctypes.c_void_p*len(holders))(*(ctypes.cast(ctypes.pointer(x),ctypes.c_void_p) for x in holders))
        if len(grid)!=3 or len(block)!=3 or min(*grid,*block)<=0:raise ValueError('Invalid dispatch geometry')
        stream=ctypes.c_void_p(torch.cuda.current_stream(self.device).cuda_stream)
        self.check(self.lib.hipModuleLaunchKernel(function,*grid,*block,shared_bytes,stream,params,None),'hipModuleLaunchKernel')
