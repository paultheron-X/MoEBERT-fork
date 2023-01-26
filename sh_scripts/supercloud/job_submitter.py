from subprocess import Popen, PIPE
import time
import numpy as np
import pandas as pd
import os
PATH = "/home/gridsan/ptheron/MoEBERT-fork"
PYPATH = "/home/gridsan/ptheron/.conda/envs/MoEBERT/bin/python"
LOG_PATH = f"{PATH}/logs"


exit_code = 1
name = "submit_hash_perm" # name of the template do not precise the .sh
ds = ['sst2']#['rte','mrpc','sst2']
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
                print(f"submitted hash routing! {ds_name} {i}")
                tmp_id = str(output)[-11:-3]
                print("job id:", tmp_id)
                submitted.append(tmp_id)
                break
            time.sleep(50000)
            
print("submitted jobs:", submitted)

#['21212106', '21212107', '21212108', '21212109', '21212110', '21212111', '21212112', '21212113', '21212114', '21212115', '21212116', '21212117', '21212118', '21212119', '21212120']

#submitted jobs: ['21213146', '21213147', '21213148', '21213149', '21213150']