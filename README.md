# Abstract
The model security has been an increasingly severe concern in the machine learning community.
Experiments show that small perturbations to the original data (i.e. adversarial attacks) could lead to
model’s misclassification. The ultimate goal of our project is to propose effective defense methods
to defend against gradient-based adversarial attacks and explore the effect of system diversity on
defense performance. Generally, the project is separated into two stages: (1) Reimplement three
classic attacks and three defenses. (2) Research on ensemble defense methods. All the attack and
defense methods will be experimented on the dataset CIFAR-10 using VGG-19 as the backbone. We
select the three defenses that we implemented as the baseline and our ensemble defense methods
finally prove to be much more effective.

Attacks methods:
(1) L-BFGS
(2) Fast Gradient Sign Method (FGSM)
(3) Universal Adversarial Attack (UAA)

Defense methods:
(1) Random Input Transformation (RIT)
(2) Random Noising (RN)
(3) FGSM Adversarial Training (FGSM AT)
(4) Ensemble-based Defense (the defense that we mainly research on)

Please refer to our group's final report to see more details.


# For testing attacks, run following three codes.
For LBFGS attack, run `python main.py LBFGS`.

For FGSM attack, run `python main.py FGSM`.

For UAA attack, run `python main.py UAA`.

# For testing defenses, run following code.
For RIT and RN, the corresponding code has been removed, because our experiments show that these two methods doesn't work for pictures with small resolution.

For FGSM-AT defense, run `python main.py FGSM-AT`.

# For testing random-ensemble methods, follow these steps.
1. Run `python random_ensemble.py`. This will give us a model pool as size of 20. The sample rate is following the report.
2. Change the directory URLs in the file `eval.py`.

    * First, in line 164, set `base_classifier_dir` to the URL of best_model.pth, the base classifier used to generate adversarial smaples.
    
    * Second, in line 165, set `left_aside_model_dir` to the URL of the non-defense model which will be used for comparison.
    
    * Third, change the list `pool_dir`, which should contain the URL of models trained from each sample rate, please follow the sample rate order `["30","40","50","60","100"]`. 
    
3. Run `python eval.py` for evaluation.



# File Structure

```bash
|____attackers
| |____attacker_base.py
| |____fgsm.py
| |____lbfgs.py
| |____uaa.py
|____defenders
| |____defender_base.py
| |____fgsm_AT.py
|____eval.py
|____main.py
|____model_files
| |____best_model.pth*
| |____model0_parameter.pkl*
| |____README.txt
|____models
| |____train.py
|____random_ensemble.py
|____README.md
|____VGG.py
```

\* **Important:** before execute our programs, you need first download the [base classifier](https://drive.google.com/drive/u/0/folders/1lapsdyaRy35wmFp6kKdNA93Saq3BC1TC), then store it under the `./model_files/` directory. Then download the [non-defense comparison model](https://drive.google.com/file/d/1-6sZEzRzHjkFUs428aXm3IxLbgKBFLZR/view?usp=sharing), then store it under the `./model_files/` directory.



# Notes

1. The first time you run our code, you may need to wait a few minutes for downlaoding the CIFAR-10 dataset.
2. The result of each attack or defense, will be printed in the console after finishing.
3. The `UAA` attack may take a long time to train (more than 12 hours). Please pay attention to that.
