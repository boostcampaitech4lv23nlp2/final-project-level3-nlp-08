import pandas as pd
import json
import os
import re

trainTypes = ['Training', 'Validation']
domains = ['개인및관계', '상거래(쇼핑)', '식음료', '여가생활', '행사']
pattern = '([ㄱ-ㅎㅏ-ㅣ]+)'
for trainType in trainTypes:
    for domain in domains:
        file_path = './data/Korean_speech_summarization/' + trainType + '/' + domain + '.json'
        with open(file_path) as f:
            data = json.load(f)

        dataLen = len(data['data'])

        X_data, y_data = [], []
        for i in range(dataLen):
            dialogue = data['data'][i]['body']['dialogue']
            summary = data['data'][i]['body']['summary']
            string = ''
            
            for j in range(len(dialogue)):
                string += dialogue[j]['utterance'] + ' '

            string = re.sub(pattern=pattern, repl='', string=string)
            X_data.append(string)
            y_data.append(summary)

    df = pd.DataFrame({'passage':X_data, 'summary':y_data})
        
    output_dir = './data_csv/' + trainType
    save_path = output_dir + '/concatenate.csv'
        
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
            
    df.to_csv(save_path)