
who : 
cse_230913008 tty2         2023-08-11 19:41 (tty2)

whoami : cse_230913008

ls 
230913008  Documents  Music     Public  Templates
Desktop    Downloads  Pictures  snap    Videos

ls -l
total 40
drwxrwxr-x 2 cse_230913008 cse_230913008 4096 Aug 11 19:59 230913008
drwxr-xr-x 2 cse_230913008 cse_230913008 4096 Aug 11 19:41 Desktop
drwxr-xr-x 2 cse_230913008 cse_230913008 4096 Aug 11 19:41 Documents
drwxr-xr-x 2 cse_230913008 cse_230913008 4096 Aug 11 19:57 Downloads
drwxr-xr-x 2 cse_230913008 cse_230913008 4096 Aug 11 19:41 Music
drwxr-xr-x 2 cse_230913008 cse_230913008 4096 Aug 11 19:41 Pictures
drwxr-xr-x 2 cse_230913008 cse_230913008 4096 Aug 11 19:41 Public
drwx------ 4 cse_230913008 cse_230913008 4096 Aug 11 20:00 snap
drwxr-xr-x 2 cse_230913008 cse_230913008 4096 Aug 11 19:41 Templates
drwxr-xr-x 2 cse_230913008 cse_230913008 4096 Aug 11 19:41 Videos


ls -a

total 40
drwxrwxr-x 2 cse_230913008 cse_230913008 4096 Aug 11 19:59 230913008
drwxr-xr-x 2 cse_230913008 cse_230913008 4096 Aug 11 19:41 Desktop
drwxr-xr-x 2 cse_230913008 cse_230913008 4096 Aug 11 19:41 Documents
drwxr-xr-x 2 cse_230913008 cse_230913008 4096 Aug 11 19:57 Downloads
drwxr-xr-x 2 cse_230913008 cse_230913008 4096 Aug 11 19:41 Music
drwxr-xr-x 2 cse_230913008 cse_230913008 4096 Aug 11 19:41 Pictures
drwxr-xr-x 2 cse_230913008 cse_230913008 4096 Aug 11 19:41 Public
drwx------ 4 cse_230913008 cse_230913008 4096 Aug 11 20:00 snap
drwxr-xr-x 2 cse_230913008 cse_230913008 4096 Aug 11 19:41 Templates
drwxr-xr-x 2 cse_230913008 cse_230913008 4096 Aug 11 19:41 Videos

pwd
/home/cse_230913008

cd 230913008/
cse_230913008@prg:~/230913008$ 

cat add.c 
#include<stdio.h>

cp:
cp add.c c-file/

ls c-file/
add.c  sub.c

mkdir test
rmdir test/
touch script.sh
chmod +x script.sh 

date : Friday 11 August 2023 08:06:39 PM IST

ls
app.py  c-file  script.sh

ls -a
.  ..  app.py  c-file  script.sh


ls -l
total 4
-rw-rw-r-- 1 cse_230913008 cse_230913008    0 Aug 11 19:59 app.py
drwxrwxr-x 2 cse_230913008 cse_230913008 4096 Aug 11 20:05 c-file
-rwxrwxr-x 1 cse_230913008 cse_230913008    0 Aug 11 20:06 script.sh

date
Friday 11 August 2023 08:08:33 PM IST

wc add.c 
 1  1 18 add.c

ps aux | grep sublime-text

cse_230+    3592  1.4  2.1 332296 83608 ?        Sl   20:00   0:09 /snap/sublime-text/122/opt/sublime_text/sublime_text --detached
cse_230+    3637  0.0  0.3 109696 14440 ?        Sl   20:00   0:00 /snap/sublime-text/122/opt/sublime_text/plugin_host-3.3 3592 /snap/sublime-text/122/opt/sublime_text/sublime_text /home/cse_230913008/.config/sublime-text /home/cse_230913008/.cache/sublime-text /snap/sublime-text/122/opt/sublime_text/Packages
cse_230+    3640  0.1  0.6 118140 23684 ?        Sl   20:00   0:00 /snap/sublime-text/122/opt/sublime_text/plugin_host-3.8 3592 /snap/sublime-text/122/opt/sublime_text/sublime_text /home/cse_230913008/.config/sublime-text /home/cse_230913008/.cache/sublime-text /snap/sublime-text/122/opt/sublime_text/Packages
cse_230+    5011  0.0  0.0   9208  2372 pts/0    S+   20:10   0:00 grep --color=auto sublime-text


sed 's/stdio/iostream/' add.c 
#include<iostream.h>


sort names.txt 
kunal
saatvik
uday
vishal

ps aux

sleep 2

sleep 10 &
[1] 5273

ps aux | grep 5273
cse_230+    5273  0.0  0.0   8368  1012 pts/0    S    20:16   0:00 sleep 10
cse_230+    5275  0.0  0.0   9076  2372 pts/0    S+   20:16   0:00 grep --color=auto 5273


kill 5275
bash: kill: (5275) - No such process
[1]+  Done                    sleep 10
