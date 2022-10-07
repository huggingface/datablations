# Permissions

Add 'umask 007` to your `.bashrc` (`~/.bashrc`) to make your new files automatically seeable and editable by the rest of the project.

You can recursively fix a brokens-perm folder by running:
```
wd=/scratch/project_462000119 #YOUR_DIRECTORY_HERE
find $wd -user `whoami` -type d ! \( -readable -executable \) -prune -o -type d -execdir chgrp project_462000119 {} + , -execdir chmod g+rwxs {} + 2>&1 | egrep -v "(Operation not permitted|cannot operate on dangling symlink)"
find $wd -user `whoami` -type d ! \( -readable -executable \) -prune -o -type f -execdir chgrp project_462000119 {} + , -execdir chmod g+rw {} + 2>&1 | egrep -v "(Operation not permitted|cannot operate on dangling symlink)"
```
