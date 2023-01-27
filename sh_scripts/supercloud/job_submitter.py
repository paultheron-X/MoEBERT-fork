from subprocess import Popen, PIPE
import time
import numpy as np
import pandas as pd
import os
PATH = "/home/gridsan/ptheron/MoEBERT-fork"
PYPATH = "/home/gridsan/ptheron/.conda/envs/MoEBERT/bin/python"
LOG_PATH = f"{PATH}/logs"


exit_code = 1
name = "submit_hash" # name of the template do not precise the .sh
ds = ['sst2']#['rte','mrpc','sst2'] #['mrpc']#['rte','mrpc','sst2']
to_run = [1, 2, 3, 4, 5]

#quit()  # to avoid running the script
submitted = []
for i in to_run:
    for ds_name in ds:
        while True:
            script_path = f"{PATH}/sh_scripts/supercloud/ds_spec_submission/hash/{name}_{ds_name}_{i}.sh"
            process = Popen(["LLsub", script_path], stdout=PIPE)        
            (output, err) = process.communicate()
            exit_code = process.wait()
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),output,err)
            if exit_code == 0:
                print(f"submitted {name} ! {ds_name} {i}")
                tmp_id = str(output)[-11:-3]
                print("job id:", tmp_id)
                submitted.append(tmp_id)
                break
            time.sleep(50000)
            
print("submitted jobs:", submitted)



#submitted jobs: ['21236632', '21236633', '21236634', '21236635', '21236636']

# hash perm ['21236712', '21236713', '21236714', '21236715', '21236716', '21236717', '21236718', '21236719', '21236720', '21236722', '21236723', '21236724', '21236725', '21236726', '21236727']

# hash sst2 ['21236843', '21236844', '21236845', '21236846', '21236847']

