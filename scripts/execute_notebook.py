from pathlib import Path
import argparse
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
parser=argparse.ArgumentParser()
parser.add_argument('--ipc',action='store_true',help='Use local IPC when TCP sockets are restricted')
parser.add_argument('--in-process',action='store_true',help='Execute through IPython without a kernel socket; trusted notebooks only')
parser.add_argument('--notebook',default='notebooks/01_fraudscope_walkthrough.ipynb')
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
path=root/args.notebook
nb=nbformat.read(path,as_version=4)
if args.in_process:
    import os
    import io
    import matplotlib.pyplot as plt
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    from IPython.display import display, Image
    os.chdir(root)
    shell=InteractiveShell.instance()
    count=0
    for cell in nb.cells:
        if cell.cell_type!='code':continue
        count+=1
        with capture_output() as captured:
            executed=shell.run_cell(cell.source,store_history=False)
            for number in plt.get_fignums():
                buf=io.BytesIO();plt.figure(number).savefig(buf,format='png',bbox_inches='tight')
                display(Image(data=buf.getvalue()));plt.close(number)
        if executed.error_before_exec:raise executed.error_before_exec
        if executed.error_in_exec:raise executed.error_in_exec
        outputs=[]
        if captured.stdout:outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
        if captured.stderr:outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
        for item in captured.outputs:outputs.append(nbformat.v4.new_output('display_data',data=item.data,metadata=item.metadata))
        cell.outputs=outputs;cell.execution_count=count
    nb.metadata['fraudscope_execution']={'method':'in-process IPython','reason':'Kernel sockets unavailable in the build environment','code_cells':count}
else:
    manager=KernelManager(kernel_name='python3',transport='ipc') if args.ipc else None
    NotebookClient(nb,km=manager,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(root)}}).execute()
nbformat.write(nb,path)
print(f'Executed {path.name}')

