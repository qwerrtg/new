import SparkApi
import asyncio
import websockets
import re
import json
from tqdm import tqdm

#以下密钥信息从控制台获取
# appid = "XXXXXXXX"     #填写控制台中获取的 APPID 信息
# api_secret = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"   #填写控制台中获取的 APISecret 信息
# api_key ="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"    #填写控制台中获取的 APIKey 信息

#用于配置大模型版本，默认“general/generalv2”
# domain = "general"   # v1.5版本
# domain = "generalv2"    # v2.0版本
domain = "generalv3"    # v3.0版本
#云端环境的服务地址
# Spark_url = "ws://spark-api.xf-yun.com/v1.1/chat"  # v1.5环境的地址
# Spark_url = "ws://spark-api.xf-yun.com/v2.1/chat" # v2.0环境的地址
Spark_url = "wss://spark-api.xf-yun.com/v3.1/chat" #v3.0环境地址

text =[]
# length = 0

def getText(role,content):
    jsoncon = {}
    jsoncon["role"] = role
    jsoncon["content"] = content
    text.append(jsoncon)
    return text

def getlength(text):
    length = 0
    for content in text:
        temp = content["content"]
        leng = len(temp)
        length += leng
    return length

def checklen(text):
    while (getlength(text) > 8000):
        del text[0]
    return text

if __name__ == '__main__':
    
    #输入数据路径
    data_input = "output.txt"
    #异常数据保存
    exception_data = "data_exception.json"
    #生成的数据
    qa_data = "qa_data.json"

    question_list = []
    pattern = "##用户：{用户：(.+)##"
    with open(data_input, "r") as file:
        lines = file.readlines()
        for line in lines:
            try:
                s = line.replace("\n","")
                match = re.search(pattern, s)
                question_list.append([match.group(1).split("##")[0],s])
            except:
                continue

    #调用星火大模型生成答案
    qa_dict = {}
    for q in tqdm(question_list):
        text=[]
        question = checklen(getText("user",q[1]))
        SparkApi.answer =""
        SparkApi.main(appid,api_key,api_secret,Spark_url,domain,question)
        answer = SparkApi.answer
        if answer=="": #如果出现异常，保存数据以便重新处理
            with open(exception_data, "a", encoding="utf-8") as f:
                ex = {}
                ex["Q"] = q[0]
                ex["data"] = q[1]
                json_obj = json.dumps(ex, ensure_ascii=False) + "\n"
                f.write(json_obj)
            continue
        else:
            #追加保存数据
            with open(qa_data, "a", encoding="utf-8") as f:
                qa = {}
                qa["input"] = q[0]
                qa["target"] = answer
                json_obj = json.dumps(qa, ensure_ascii=False) + "\n"
                f.write(json_obj)
            continue
