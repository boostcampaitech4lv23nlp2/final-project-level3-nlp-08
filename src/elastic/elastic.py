from elasticsearch import Elasticsearch, helpers
from typing import Optional, Dict
import json
import warnings
from glob import glob
import os
import random

warnings.filterwarnings("ignore")

class ElasticObject:
    
    summary_messages = ["지금까지 대화한 내용을 요약해 봤어!", "멍멍멍멍멍멍(대충 요약했다는 뜻)", "너네 이런 대화했지? 맞췄지? 잘했지?"]
    recommend_messages = ["내가 추천해 주는 글이 도움이 될거야!", "내가 열심히 찾아봤다 멍멍!", "이게 좋겠다!", "멍머멍(대충 문서를 가져왔다는 뜻)"]
    
    def __init__(self, host: str, port: Optional[str] = None) -> None:
        """
        엘라스틱서치 커넥션 클래스

        Args:
            host (str): 엘라스틱서치 호스트
            port (str): 엘라스틱서치 포트
        """

        self.host = host
        self.port = port

        if not self.host.startswith("http"):
            self.host = "http://" + self.host

        if self.port:
            self.host = self.host + ":" + self.port

        self._connect_server(self.host)

    def _connect_server(self, url: str):
        """
        엘라스틱서치 서버와 연결

        Args:
            url (str): 엘라스틱서치 URL

        """

        self.client = Elasticsearch(
            url, timeout=30, max_retries=10, retry_on_timeout=True
        )
        print(f"Connected to Elastic Server ({url})")

    def create_index(self, index_name: str, setting_path: str = "./settings.json"):
        """_summary_

        Args:
            index_name (str): Name of an index
            setting_path (str): Path of the setting file
        """

        with open(setting_path, encoding="utf-8") as f:
            settings = json.load(f)

        if self.client.indices.exists(index=index_name):
            print(f"{index_name} already exists.")
            usr_input = input("Do you want to delete? (Y/n)")
            if usr_input == "Y":
                self.client.indices.delete(index=index_name)

            else:
                return False

        self.client.indices.create(index=index_name, body=settings)
        print(f"Create an Index ({index_name})")
        return True

    def get_indices(self):
        indices = list(self.client.indices.get_alias().keys())
        return indices

    def delete_index(self, index_name: str):
        """_summary_

        Args:
            index_name (str): Name of the index
        """
        if self.client.indices.exists(index=index_name):
            self.client.indices.delete(index=index_name)
            print(f"Delete an Index ({index_name})")

        else:
            print(f"Not exist {index_name}")

    def insert_data(
        self,
        index_name: str,
        data_path: str,
    ):
        """_summary_

        Args:
            index_name (str): Name of an index
            data_path (str): Path of the Document file(json) or dir
        """
        if os.path.isdir(data_path):
            data_list = glob(data_path + '/*.json')
        
        else:
            data_list = [data_path]
        
        docs = []
        i = self.document_count(index_name=index_name)
        for data_path in data_list:
            with open(data_path, encoding="utf-8") as f:
                data = json.load(f)

            print("Data Loding...")
            for v in data['posts']:
                if not self._check_docs(url=v['url'], index_name=index_name):
                    doc = {
                        "_index": index_name,
                        "_type": "_doc",
                        "_id": i+1,
                        "title": v["title"],
                        "context": v["content"],
                        "url": v['url'],
                        "copyright": v['copyright'],
                        "like": int(v['like']) if v['like'] else 0
                        
                    }
                    docs.append(doc)
                    i += 1

        helpers.bulk(self.client, docs)

        print("Data Upload Completed")
        self.document_count(index_name)
        
    def _check_docs(self, url, index_name):
        body = {
            "query": {
                "term": {
                    "url.keyword": url
                }
            }
        }
        
        return bool(self.client.search(index=index_name, body=body)['hits']['total']['value'])
        

    def delete_data(self, index_name: str, doc_id):
        """_summary_

        Args:
            index_name (_type_): _description_
            doc_id (_type_): _description_
        """

        self.client.delete(index=index_name, id=doc_id)

        print(f"Deleted {doc_id} document.")
        
    def init_index(self, index_name: str):
        if self.client.indices.exists(index=index_name):
            self.delete_index(index_name=index_name)
            
        self.create_index(index_name=index_name)
        print(f"Initialization...({index_name})")

    def document_count(self, index_name: str):

        counts = self.client.count(index=index_name, pretty=True)["count"]
        print(f"Number of documents to {index_name} is {counts}.")
        return counts

    def search(self, index_name: str, question: str, topk: int = 10):

        body = {
            "_source": {
                "includes": ["title", "url", "like"]
            }
            ,
            "query": 
                {
                    "bool": 
                        {"must": 
                            [
                                {"match": 
                                    {
                                        "content": question
                                        }
                                    }
                                ]
                            }
                        }
                }

        responses = self.client.search(index=index_name, body=body, size=topk)["hits"]["hits"]
        responses=sorted(responses, key=lambda x:-int(x['_source']['like']))
        
        
        random_summary_idx = random.randint(0, len(self.summary_messages)-1)
        random_recommend_idx = random.randint(0, len(self.recommend_messages)-1)
        
        api_output = {
            "location": "recommend",
            "summary_message": self.summary_messages[random_summary_idx],
            "summary": question,
            "recommend_message": self.recommend_messages[random_recommend_idx],
            "source": responses
        }
        
        return responses, api_output


if __name__ == "__main__":

    es = ElasticObject("localhost:9200")
    # es.create_index('blogs', setting_path='./settings.json')
    
    outputs = es.search('blogs', "여수시수여수 여수의 여수는 여수여수와 여수를 함께 하고 있다.")
    print(outputs)
    
        
        
