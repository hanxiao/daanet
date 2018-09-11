# DAANet: Dual Ask-Answer Network for Machine Reading Comprehension
[![Python: 3.6](https://img.shields.io/badge/Python-3.6-brightgreen.svg)](https://opensource.org/licenses/MIT)    [![Tensorflow: 1.6](https://img.shields.io/badge/Tensorflow-1.6-brightgreen.svg)](https://opensource.org/licenses/MIT)  


![](.github/a78d34ce.png)

## Requirements

- Python >= 3.6
- Tensorflow >= 1.6 (self-compiled TF-gpu is recommended!)
- gputil >= 1.3.0
- GPU

## Usage
For dual learning run:
```
python grid_search.py daanet 
```
For QA-only model (corresponds to `mono` in the experiment) run: 
```
python grid_search.py monoqa 
```
For QG-only model (corresponds to `mono` in the experiment) , run :
```
python grid_search.py monoqg 
```

All hyperparameters used in the paper are stored in `default.yaml`:

You can change the data path and save dir in `grid.yaml`


## Evaluation

Evaluation on the dev set is automatically done after each epoch.

To do evaluation manually,

```bash
python app.py evaluate ~/save/models/DDMM-HHMMSS/default.yaml
```

, where `~/save/models/DDMM-HHMMSS/default.yaml` is the saved yaml config of model `DDMM-HHMMSS`. It is created during the training procedure. It automatically loads the parameters from the best epoch (or fallback to the last epoch) to the model.


## Continuous Training
```bash
python app.py train ~/save/models/DDMM-HHMMSS/default.yaml
```
It will load the best (or last) model so far and conducts incremental training.

## Generated Samples
Selected outputs from DAANET and mono. Yellow text is question-related; green text is answer-related.

![](.github/9f38cfd8.png)
![](.github/859b252b.png)
![](.github/0d4f4707.png)
![](.github/0355fc42.png)
![](.github/7145b5b9.png)
![](.github/f71f7ecd.png)
![](.github/edd2517e.png)


## Attention Matrix
Question-Context and Answer Context attention matrices

![](.github/f951034d.png)
![](.github/34ef53b5.png)
![](.github/bbbf5483.png)
![](.github/9e0dcdf7.png)
![](.github/e48e682e.png)
![](.github/12032683.png)
![](.github/d7033a00.png)
![](.github/390adbc3.png)
