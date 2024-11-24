#!/usr/bin/env python
# coding: utf-8


from langchain.schema import Document
from langchain.vectorstores import Chroma,FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from pdf_parse import DataProcess
import torch


class FaissRetriever(object):
    # 初始化文档块索引，然后插入faiss库
    def __init__(self, model_path, data):
        self.embeddings  = HuggingFaceEmbeddings(
                               model_name = model_path,
                               model_kwargs = {"device":"cuda"}
                               # model_kwargs = {"device":"cuda:1"}
                           )
        docs = []
        for idx, line in enumerate(data):
            line = line.strip("\n").strip()
            words = line.split("\t")
            docs.append(Document(page_content=words[0], metadata={"id": idx}))
        self.vector_store = FAISS.from_documents(docs, self.embeddings)
        del self.embeddings
        torch.cuda.empty_cache()

    # 获取top-K分数最高的文档块
    def GetTopK(self, query, k):
       context = self.vector_store.similarity_search_with_score(query, k=k)
       return context

    # 返回faiss向量检索对象
    def GetvectorStore(self):
        return self.vector_store

if __name__ == "__main__":
    base = "."
    model_name = base + "/pre_train_model/m3e-large" #text2vec-large-chinese
    dp =  DataProcess(pdf_path = base + "/data/train_a.pdf")
    dp.ParseBlock(max_seq = 1024)
    dp.ParseBlock(max_seq = 512)
    print(len(dp.data))
    dp.ParseAllPage(max_seq = 256)
    dp.ParseAllPage(max_seq = 512)
    print(len(dp.data))
    dp.ParseOnePageWithRule(max_seq = 256)
    dp.ParseOnePageWithRule(max_seq = 512)
    print(len(dp.data))
    data = dp.data

    faissretriever = FaissRetriever(model_name, data)
    faiss_ans = faissretriever.GetTopK("如何预防新冠肺炎", 6)
    print(faiss_ans)
    faiss_ans = faissretriever.GetTopK("交通事故如何处理", 6)
    print(faiss_ans)
    faiss_ans = faissretriever.GetTopK("吉利集团的董事长是谁", 6)
    print(faiss_ans)
    faiss_ans = faissretriever.GetTopK("吉利汽车语音组手叫什么", 6)
    print(faiss_ans)
