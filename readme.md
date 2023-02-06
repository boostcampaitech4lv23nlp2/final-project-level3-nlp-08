# Final Project NLP-08

# monstache

```terminal
$ wget https://github.com/rwynn/monstache/releases/download/v6.7.4/monstache-98f8bc6.zip

$ unzip monstache-98f8bc6.zip
```

```bash
$ cd ~
$ mkdir monstache
$ cd monstache
```
`vi mongo-elastic.toml`

```toml
# connection settings

# connect to MongoDB using the following URL
mongo-url = "mongodb+srv://nlp-08:finalproject@cluster0.rhr2bl2.mongodb.net/?retryWrites=true&w=majority"

# connect to the Elasticsearch REST API at the following node URLs
elasticsearch-urls = ["http://localhost:9200"]

# frequently required settings
# if you need to seed an index from a collection and not just listen and sync changes events
# you can copy entire collections or views from MongoDB to Elasticsearch
direct-read-namespaces = ["test_database.blogs"]

# if you want to use MongoDB change streams instead of legacy oplog tailing use change-stream-namespaces
# change streams require at least MongoDB API 3.6+
# if you have MongoDB 4+ you can listen for changes to an entire database or entire deployment
# in this case you usually don't need regexes in your config to filter collections unless you target the deployment.
# to listen to an entire db use only the database name.  For a deployment use an empty string.
change-stream-namespaces = ["test_database.blogs"]

# additional settings
# compress requests to Elasticsearch
gzip = true

# generate indexing statistics
stats = true

# index statistics into Elasticsearch
index-stats = true

# use 4 go routines concurrently pushing documents to Elasticsearch
elasticsearch-max-conns = 4

# propogate dropped collections in MongoDB as index deletes in Elasticsearch
dropped-collections = false

# propogate dropped databases in MongoDB as index deletes in Elasticsearch
dropped-databases = false

# in Elasticsearch with a newer version. Elasticsearch is preventing the old docs from overwriting new ones.
replay = false

# resume processing from a timestamp saved in a previous run
resume = false

# do not validate that progress timestamps have been saved
resume-write-unsafe = true

# override the name under which resume state is saved
resume-name = "default"

# use a custom resume strategy (tokens) instead of the default strategy (timestamps)
# tokens work with MongoDB API 3.6+ while timestamps work only with MongoDB API 4.0+
resume-strategy = 1

# print detailed information including request traces
verbose = true
index-as-update = true
index-oplog-time = false
index-files = false
file-highlighting = false

[[mapping]]
namespace = "test_database.blogs"
index = "blogs"
```

```
monstache -f /mongo-elastic.toml
```

background로 돌리고 싶을때 `nohup`사용

## develop_summary용 How to run

### Train
```commandLine
python train.py --config {config}
```
- config옵션을 주지 않을 경우, config/base.yaml이 실행됨.
- 실험 환경 세팅은 yaml 파일을 변경함.

### Inference
```commandLine
python test.py --config {config} --data {data_path} --model_path {model_path}
```
- config 옵션을 주지 않을 경우, config/base.yaml이 실행됨.
- data 옵션을 주지 않을 경우, config/base.yaml의 predict_path를 불러옴.
- model_path 옵션을 주지 않을 경우, config/base.yaml의 model_path를 불러옴.
