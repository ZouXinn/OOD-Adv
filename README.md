# OOD-Adv

## On the Adversarial Robustness of Out-of-distribution Generalization Models

This code is used to reproduce the experiments the main paper of:


**On the Adversarial Robustness of Out-of-distribution Generalization Models**
Xin Zou, Weiwei Liu
37th Conference on Neural Information Processing Systems (NeurIPS), 2023
[[Website]](./not-avaliable-yet) [[Paper]](./not-avaliable-yet)


Our code is based on **DomainBed**, for more details about the benchmark DomainBed, please refer to [DomainBed](https://github.com/facebookresearch/DomainBed).

# Reproduce Experimental Results

## 1 Download the datasets
```sh
python -m domainbed.scripts.download --data_dir=./domainbed/data
```

## 2 Lunch the sweeps
Lunch the sweep for the algorithms that does not use adversarial training:
```sh
python -m domainbed.scripts.sweep launch --data_dir=./domainbed/data \
            --output_dir=./sweep/output/path \
            --command_launcher multi_gpu \
            --algorithms ERM MLDG CDANN VREx RSC \
            --datasets RotatedMNIST ColoredMNIST VLCS PACS OfficeHome \
            --n_hparams 20 --n_trials 1 --single_test_envs
```

Lunch the sweep for MAT and LDAT:
```sh
python -m domainbed.scripts.sweep launch --data_dir=./domainbed/data \
            --output_dir=./sweep/output/path \
            --command_launcher multi_gpu \
            --algorithms MAT LDAT \
            --datasets ColoredMNIST VLCS PACS OfficeHome \
            --n_hparams 20 --n_trials 1 --single_test_envs \
            --steps 8000
```

Lunch the sweep for AERM and RDANN:
```sh
python -m domainbed.scripts.sweep launch --data_dir=./domainbed/data \
            --output_dir=./sweep/output/path \
            --command_launcher multi_gpu \
            --algorithms AERM RDANN \
            --datasets RotatedMNIST ColoredMNIST VLCS PACS OfficeHome \
            --n_hparams 20 --n_trials 1 --single_test_envs \
            --steps 8000
```

## 3 find the best parameters
```sh
python -m domainbed.scripts.find_best_hparam --input_dir=./sweep/output/path \
            --dataset=RotatedMNIST

python -m domainbed.scripts.find_best_hparam --input_dir=./sweep/output/path \
            --dataset=ColoredMNIST

python -m domainbed.scripts.find_best_hparam --input_dir=./sweep/output/path \
            --dataset=VLCS
            
python -m domainbed.scripts.find_best_hparam --input_dir=./sweep/output/path \
            --dataset=PACS
            
python -m domainbed.scripts.find_best_hparam --input_dir=./sweep/output/path \
            --dataset=OfficeHome
```

## 4 Retrain the models with the best parameters
```sh
python -m domainbed.scripts.resweep launch --command_launcher multi_gpu \
            --datasets RotatedMNIST ColoredMNIST VLCS PACS OfficeHome \
            --algorithms ERM MLDG CDANN VREx RSC MAT LDAT AERM RDANN \
            --sweep_base_dir ./sweep/output \
            --selection_methods IIDAccuracySelectionMethod
```

## 5 Attack the trained models
```sh
python -m domainbed.scripts.sweep_attack launch --command_launcher multi_gpu \
            --datasets RotatedMNIST ColoredMNIST VLCS PACS OfficeHome \
            --sweep_base_dir ./sweep/output \
            --selection_methods IIDAccuracySelectionMethod \
            --train_methods ST \
            --attacks FGSM PGD \
            --algorithms ERM MLDG CDANN VREx RSC MAT LDAT AERM RDANN

python -m domainbed.scripts.sweep_attack launch --command_launcher multi_gpu \
            --datasets RotatedMNIST VLCS PACS OfficeHome \
            --sweep_base_dir ./sweep/output \
            --selection_methods IIDAccuracySelectionMethod \
            --train_methods ST \
            --attacks AutoAttack \
            --algorithms ERM MLDG CDANN VREx RSC MAT LDAT AERM RDANN
```

## 6 Collect the results
Run the following command to collect the results for FGSM and PGD
```sh
python -m domainbed.scripts.collect_attack_results --input_dir ./sweep/output \
            --datasets RotatedMNIST ColoredMNIST VLCS PACS OfficeHome \
            --selection_methods IIDAccuracySelectionMethod \
            --train_method ST \
            --attacks FGSM PGD \
            --algorithms ERM MLDG CDANN VREx RSC MAT LDAT AERM RDANN \
            --attack_all_in_one --latex
```

**Then, you can find the .tex result file in the "./OOD-Adv/sweep/output/attacks_results/" folder.**

Run the following command to collect the results for AutoAttack
```sh
python -m domainbed.scripts.collect_attack_results --input_dir ./sweep/output \
            --datasets RotatedMNIST VLCS PACS OfficeHome \
            --selection_methods IIDAccuracySelectionMethod \
            --train_method ST \
            --attacks AutoAttack \
            --algorithms ERM MLDG CDANN VREx RSC MAT LDAT AERM RDANN \
            --attack_all_in_one --latex
```
**Then, you can find the .tex result file in the "./OOD-Adv/sweep/output/attacks_results/" folder.**

Note that the result file will be overwritten if you run the two commands sequentially.