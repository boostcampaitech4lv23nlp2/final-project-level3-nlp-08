# 실시간 대화 요약 & 컨텐츠 추천 봇: TravelTalk

## 1. Introduction

프로젝트 설명


#### Members

|                            김현수                            |                            이성구                            |                            이현준                            |                            조문기                            |                            조익노                            |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src='https://avatars.githubusercontent.com/u/97166760?v=4' height=80 width=80px></img> | <img src='https://avatars.githubusercontent.com/u/33012030?v=4' height=80 width=80px></img> | <img src='https://avatars.githubusercontent.com/u/87929279?v=4' height=80 width=80px></img> | <img src='https://avatars.githubusercontent.com/u/28976334?v=4' height=80 width=80px></img> | <img src='https://avatars.githubusercontent.com/u/49403520?v=4' height=80 width=80px></img> |
| [![Git Badge](http://img.shields.io/badge/-Github-black?style=flat-square&logo=github)](https://github.com/khs0415p) | [![Git Badge](http://img.shields.io/badge/-Github-black?style=flat-square&logo=github)](https://github.com/papari1123) | [![Git Badge](http://img.shields.io/badge/-Github-black?style=flat-square&logo=github)](https://github.com/coderJoon) | [![Git Badge](http://img.shields.io/badge/-Github-black?style=flat-square&logo=github)](https://github.com/siryuon) | [![Git Badge](http://img.shields.io/badge/-Github-black?style=flat-square&logo=github)](https://github.com/iknocho) |

### Contribution

- [`김현수`](https://github.com/khs0415p) &nbsp; Project Management • Service Dataset • Front-end & Back-end Update • EDA
- [`이성구`](https://github.com/papari1123) &nbsp; Modeling • Model Optimization • AutoML • EDA
- [`이현준`](https://github.com/coderJoon) &nbsp; Model Optimization • Application Cloud Release (GKE) • Service Architecture
- [`조문기`](https://github.com/siryuon) &nbsp; Baseline Code • Modeling • Model Optimization • EDA
- [`조익노`](https://github.com/iknocho) &nbsp; Service Dataset • EDA • Front-end & Back-end Update


## 2. Model

### KoBART

### ColBERT


## 3. Flow Chart

### System Architecture

Image

### Pipeline

Image

## 4. How to Use

### Install Requirements

```bash
pip install -r requirements.txt
```

### Project Tree

```
|-- app
|   |-- assets
|   |-- src
|   |   |-- elastic
|   |   └-- models
|   |-- templates
|   |-- app.py
|   |-- config.py
|   └-- mongodb.py
|-- train
|   |-- summary
|   └-- retriever
└-- monstache
    └-- mongo-elastic.toml

```

### Getting Started
- Train
```python
python train.py -c config.json
```
- Run
```python
python test.py -c config.json    # test_config.json
```


## 5. Demo 

## 6. Reference
