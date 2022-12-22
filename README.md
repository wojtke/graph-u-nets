# Graph U-nets for classification
### Split dataset 
Run `split_dataset.py` to split selected dataset and save indices.
```
usage: python split_dataset.py [options]

optional arguments:
  --dataset DATASET         Dataset name
  --name NAME               Split name (default: 'default')
  --method {cv,holdout}     Split method
  --folds FOLDS             Number of folds for cross-validation (default 10)
  --test-size TEST_SIZE     Test size for holdout
  --seed SEED               Seed for reproducibility (default 0)
```


### End to end model selection and assessment 
Run `main.py` to perform an experiment.

```
usage: python main.py [options]

optional arguments:                                                                                                                                                                                                                                                                                                                                                                          
  --dataset DATASET     dataset                                                                                                                                                                                              
  --split SPLIT         Split name (default 'default')                                                                                                                                                                                        
  --pool POOL           Pooling method (default topk)                                                                                                                                                                                     
  --conv CONV           Convolution method (default gcn)                                                                                                                                                                              
  --selection-trials SELECTION_TRIALS                                                                                                                                                                                        
                        Number of trials for hyperparameter selection (default 50)                                                                                                                                                       
  --test-trials TEST_TRIALS                                                                                                                                                                                                  
                        Number of trials for model testing (default 3)                                                                                                                                                                
  --hyperparams-space HYPERPARAMS_SPACE                                                                                                                                                                                      
  --n-jobs N_JOBS       Number of parallel jobs for hyperparameter selection (default 1)                                                                                                                                              
  --seed SEED           Seed for reproducibility                                                                                                                                                                             
  --device DEVICE       Device to be used for training  
```
