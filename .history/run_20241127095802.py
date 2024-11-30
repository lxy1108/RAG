#!/usr/bin/env python
# coding: utf-8

import json
import jieba
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
from langchain.schema import Document
from langchain.vectorstores import Chroma,FAISS
from langchain import PromptTemplate, LLMChain
from langchain.chains import RetrievalQA
import time
import re

from vllm_model import ChatLLM, Baichuan
from rerank_model import reRankLLM
from retrievers.faiss_retriever import FaissRetriever
from retrievers.bm25_retriever import BM25
from retrievers.tfidf_retriever import TFIDF
from pdf_parse import DataProcess
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1,2"# 获取Langchain的工具链 
os.environ["NCCL_IGNORE_DISABLED_P2P"] = '1'
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

def get_qa_chain(llm, vector_store, prompt_template):

    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])

    return RetrievalQA.from_llm(llm=llm, retriever=vector_store.as_retriever(search_kwargs={"k": 10}), prompt=prompt)

def retrievers_recall(retrievers, query, topk=15, maxnum=6):
    merged_docs = []
    doc_set = set()
    for retriever in retrievers:
        cnt = 0
        docs = retriever.GetTopK(query, topk)
        for doc in docs:
            if isinstance(doc, tuple):
                doc = doc[0]
            cnt += 1
            if(cnt>maxnum):
                break
            if doc.metadata['id'] in doc_set:
                continue
            merged_docs.append(doc)
            doc_set.add(doc.metadata['id'])
    return merged_docs

def reRank(rerank, top_k, query, docs):
    rerank_ans = rerank.predict(query, docs)
    rerank_ans = rerank_ans[:top_k]
    # docs_sort = sorted(rerank_ans, key = lambda x:x.metadata["id"])
    emb_ans = ""
    
    for doc in rerank_ans:
        if(len(emb_ans + doc.page_content) > max_length):
            break
        emb_ans = emb_ans + doc.page_content
    return emb_ans         
            
    

# 构造提示，根据merged faiss和bm25的召回结果返回答案
def get_emb_bm25_merge(faiss_context, bm25_context, query):
    max_length = 2500
    emb_ans = ""
    cnt = 0
    for doc, score in faiss_context:
        cnt =cnt + 1
        if(cnt>6):
            break
        if(len(emb_ans + doc.page_content) > max_length):
            break
        emb_ans = emb_ans + doc.page_content
    bm25_ans = ""
    cnt = 0
    for doc in bm25_context:
        cnt = cnt + 1
        if(len(bm25_ans + doc.page_content) > max_length):
            break
        bm25_ans = bm25_ans + doc.page_content
        if(cnt > 6):
            break

    prompt_template = """基于以下已知信息，简洁和专业的来回答用户的问题。
                                如果无法从中得到答案，请说 "无答案"，不允许在答案中添加编造成分，答案请使用中文。
                                已知内容为吉利控股集团汽车销售有限公司的吉利用户手册:
                                1: {emb_ans}
                                2: {bm25_ans}
                                问题:
                                {question}""".format(emb_ans=emb_ans, bm25_ans = bm25_ans, question = query)
    return prompt_template


def get_rerank(emb_ans, query):

    prompt_template = """基于以下已知信息，简洁和专业的来回答用户的问题。
                                如果无法从中得到答案，请说 "无答案" ，不允许在答案中添加编造成分，答案请使用中文。
                                已知内容为吉利控股集团汽车销售有限公司的吉利用户手册:
                                1: {emb_ans}
                                问题:
                                {question}""".format(emb_ans=emb_ans, question = query)
    return prompt_template


def question(text, llm, vector_store, prompt_template):

    chain = get_qa_chain(llm, vector_store, prompt_template)

    response = chain({"query": text})
    return response

# def reRank(rerank, top_k, query, bm25_ans, faiss_ans):
#     items = []
#     max_length = 4000
#     for doc, score in faiss_ans:
#         items.append(doc)
#     items.extend(bm25_ans)
#     rerank_ans = rerank.predict(query, items)
#     rerank_ans = rerank_ans[:top_k]
#     # docs_sort = sorted(rerank_ans, key = lambda x:x.metadata["id"])
#     emb_ans = ""
#     for doc in rerank_ans:
#         if(len(emb_ans + doc.page_content) > max_length):
#             break
#         emb_ans = emb_ans + doc.page_content
#     return emb_ans

# 解析pdf文档，构造数据
def parse_pdf(file):
    dp =  DataProcess(pdf_path = file)
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
    print("data load ok")
    return data
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='My RAG System')
    
    # file paths
    parser.add_argument('--document_path', type=str, required=False, default='data/train_a.pdf', help='Document path')
    parser.add_argument('--question_path', type=str, required=False, default='data/test_question.json', help='Question path')
    parser.add_argument('--output_path', type=str, required=False, default='data/result1.json', help='Output answer path')

    # model names
    parser.add_argument('--dense_retriever_lists', type=str, required=False, default='m3e,bge,gte,bce"', help='Dense retrieve model lists')
    parser.add_argument('--reranker_name', type=str, required=False, default='bce', help='Rerank model name')
    parser.add_argument('--llm_name', type=str, required=False, default='qwen', help='LLM model name')

    # hyper-parameters
    parser.add_argument('--rerank_topk', type=int, required=False, default='pre_train_model/Qwen1.5-7B-Chat', help='LLM model name')

    args = parser.parse_args()

    start = time.time()
    
    data =  parse_pdf(args.document_path)
    
    dense_retriever_names = args.dense_retriever_lists.split(',')
    retrievers = []
    for embed_model in args.dense_retriever_lists.split(','):
        retrievers.append(FaissRetriever(embed_model, data))
    print("faissretriever load ok")
    
    # BM25召回
    retrievers.append(BM25(data))
    print("bm25 load ok")
    
    # TFIDF召回
    retrievers.append(TFIDF(data))
    print("tfidf load ok")

    reranker_dict = {
        "bge": "pre_train_model/bge-reranker-large",
        "bce": "pre_train_model/bce-reranker-base"
    }
    assert args.reranker_name in reranker_dict.keys(), "Unknown rerank model name: {args.reranker_name}."
    
    # reRank模型
    rerank = reRankLLM(reranker_dict[args.reranker_name])
    print("rerank model load ok")
    
    llm_dict = {
        "qwen": (ChatLLM, "pre_train_model/Qwen1.5-7B-Chat"),
        "baichuan": (ChatLLM, "pre_train_model/Baichuan2-7B-Chat"),
        "chatglm": (Baichuan, "pre_train_model/chatglm3-6b")
    }
    assert args.llm_name in llm_dict.keys(), "Unknown LLM model name: {args.llm_name}."
    
    # LLM大模型
    llm_model, llm_path = llm_dict[args.llm_name]
    llm = llm_model(llm_path)
    print("llm {args.llm_name} load ok")

    

    # 对每一条测试问题，做答案生成处理
    with open(args.question, "r" , encoding="utf-8") as f:
        jdata = json.loads(f.read())
        print(len(jdata))
        max_length = 4000
        for idx, line in enumerate(jdata):
            query = line["question"]

            # faiss召回topk
            # faiss_context = faissretriever.GetTopK(query, 15)
            # faiss_min_score = 0.0
            # if(len(faiss_context) > 0):
            #     faiss_min_score = faiss_context[0][1]
            # cnt = 0
            # emb_ans = ""
            # for doc, score in faiss_context:
            #     cnt =cnt + 1
            #     # 最长选择max length
            #     if(len(emb_ans + doc.page_content) > max_length):
            #         break
            #     emb_ans = emb_ans + doc.page_content
            #     # 最多选择6个
            #     if(cnt>6):
            #         break

            # # bm2.5召回topk
            # bm25_context = bm25.GetBM25TopK(query, 15)
            # bm25_ans = ""
            # cnt = 0
            # for doc in bm25_context:
            #     cnt = cnt + 1
            #     if(len(bm25_ans + doc.page_content) > max_length):
            #         break
            #     bm25_ans = bm25_ans + doc.page_content
            #     if(cnt > 6):
            #         break

            # # 构造合并bm25召回和向量召回的prompt
            # emb_bm25_merge_inputs = get_emb_bm25_merge(faiss_context, bm25_context, query)

            # # 构造bm25召回的prompt
            # bm25_inputs = get_rerank(bm25_ans, query)

            # # 构造向量召回的prompt
            # emb_inputs = get_rerank(emb_ans, query)

            # # rerank召回的候选，并按照相关性得分排序
            # rerank_ans = reRank(rerank, 6, query, bm25_context, faiss_context)
            # # 构造得到rerank后生成答案的prompt
            # rerank_inputs = get_rerank(rerank_ans, query)
            
            docs = retrievers_recall(retrievers, query)
            context = reRank(rerank, 6, query, docs)
            inputs = get_rerank(context, query)

            batch_input = [inputs]
            # batch_input.append(emb_bm25_merge_inputs)
            # batch_input.append(bm25_inputs)
            # batch_input.append(emb_inputs)
            # batch_input.append(rerank_inputs)
            # 执行batch推理
            batch_output = llm.infer(batch_input)
            line["answer"] = batch_output[0] # 合并两路召回的结果
            # line["answer_2"] = batch_output[1] # bm召回的结果
            # line["answer_3"] = batch_output[2] # 向量召回的结果
            # line["answer_4"] = batch_output[3] # 多路召回重排序后的结果
            # line["answer_5"] = emb_ans
            # line["answer_6"] = bm25_ans
            # line["answer_7"] = rerank_ans
            # 如果faiss检索跟query的距离高于500，输出无答案
            # if(faiss_min_score >500):
            #     line["answer_5"] = "无答案"
            # else:
            #     line["answer_5"] = str(faiss_min_score)

        # 保存结果，生成submission文件
        json.dump(jdata, open(args.output_path, "w", encoding='utf-8'), ensure_ascii=False, indent=2)
        end = time.time()
        print("cost time: " + str(int(end-start)/60))
