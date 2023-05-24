import os
import shutil
# shutil.rmtree()

checkpoint_dirs = [dir_name for dir_name in os.listdir() if dir_name.startswith('checkpoint')]

for dir_name in checkpoint_dirs:
    latest_file_path = os.path.join(dir_name, 'latest')
    with open(latest_file_path, 'r') as f:
        latest_checkpoint = f.read().strip()
    if not os.path.exists(os.path.join(dir_name, latest_checkpoint)):
        print(f"Deleting directory {dir_name} because checkpoint {latest_checkpoint} does not exist in it.")
        shutil.rmtree(dir_name)
        #break
        #os.rmdir(dir_name)
