import platform
import os
import shutil
import femm


def myopen(smart_flag=1):
    if platform.system().lower() == 'windows':
        femm.openfemm(1)
    else:
        femm.openfemm(winepath=os.environ["WINEPATH"], femmpath=os.environ["FEMMPATH"])
    femm.smartmesh(smart_flag)

    
def cleanup_femm_files(clean_dir, move_dir=None):
    files = os.listdir(clean_dir)
    for f in files:
        if f.endswith(".ans") or f.endswith(".fem") or f.endswith(".csv"):
            if move_dir is None:
                os.remove(os.path.join(clean_dir, f))
            else:
                shutil.move(os.path.join(clean_dir, f), os.path.join(move_dir, f))
