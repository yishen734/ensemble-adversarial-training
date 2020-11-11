# How to use
Basically, one line of code could work.



## For testing attacks, run following three codes.
For LBFGS attack, run "python main.py LBFGS".
  
For FGSM attack, run "python main.py FGSM".
 
For UAA attack, run "python main.py UAA".

## For testing defenses, run following code.
For FGSM-AT defense, run "python main.py FGSM-AT".


##Notes
First download model from "https://drive.google.com/drive/u/0/folders/1lapsdyaRy35wmFp6kKdNA93Saq3BC1TC", then store it under "model_files" directory.

The first time you run our code, you may need to wait a few minutes for downlaoding the CIFAR-10 dataset.

The result of each attack or defense, will be printed in the console after finishing.

The "UAA" attack may take a long time to train (more than 12 hours). Please pay attention to that.

The whole file structure is as following:

--GRP  
-----attackers  
--------attacker_base.py  
--------fgsm.py  
--------lbfgs.py  
--------uaa.py  
-----data  
-----defenders  
--------defender_base.py  
--------fgsm_AT.py  
-----model_files
--------best_model.pth  
-----main.py  
-----README.md  
-----VGG.py
