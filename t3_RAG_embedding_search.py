""" test OpenAI building of prompts using proximity search on related documents """
# imports
import ast  # for converting embeddings saved as strings back to arrays
from openai import OpenAI # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import numpy as np    #for faiss

import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search
from transformers import GPT2Tokenizer
import faiss

from datetime import  datetime
import os
import shutil
import mwparserfromhell

# utilities from py_utz
from utz import utz

class test:
    EMBEDDING_MODEL = "text-embedding-ada-002"
    # EMBEDDING_MODEL = "text-embedding-3-large"  # has too many dimensions compared to ada-002
    # EMBEDDING_MODEL = "text-embedding-3-small"  # same dimansions but distance reversed  compared to ada-002

    # GPT_MODEL = "gpt-3.5-turbo"
    # GPT_MODEL = "gpt-4"
    GPT_MODEL = "gpt-4-turbo-preview"

    def __init__(self) -> None:
        utz.logset()
        self.client = OpenAI()
        self.token_budget = 4096 - 500
        # self.token_budget =  10**5
        self.df_file="catalog/df.file"
        self.df_file_split="catalog/df_split.file"
        self.docs_dir = 'catalog/docs'
        self.faiss_fn="catalog/faiss.index"
        pass


    @utz.time_decorator
    def embedding_get_from_samples(self):
        # download pre-chunked text and pre-computed embeddings
        # this file is ~200 MB, so may take a minute depending on your connection speed
        """get list of embedding for wiki from external site and save it"""
        embeddings_path = "https://cdn.openai.com/API/examples/data/winter_olympics_2022.csv"
        self.df = pd.read_csv(embeddings_path)

        # convert embeddings from CSV str type back to list type
        self.df['embedding'] = self.df['embedding'].apply(ast.literal_eval)
        utz.print("len of embedding df ", len(self.df))
        self.df.to_pickle(self.df_file)
        pass

    def util_get_doc_filename(self,file_id):
        return self.docs_dir +"/" +str(file_id).zfill(5)

    @utz.time_decorator
    def embedding_split(self):
        """ split df . move text column to a folder with files"""
        self.df=pd.read_pickle(self.df_file)
        self.df_split=pd.DataFrame({"embedding": self.df["embedding"].copy(),"file_id":self.df.index})
        self.df_split.to_pickle(self.df_file_split)

        # Re-create the doc folder
        if os.path.exists(self.docs_dir):
            shutil.rmtree(self.docs_dir)
        os.makedirs(self.docs_dir)
        for file_id, doc in self.df["text"].items():
            with open(self.util_get_doc_filename(file_id), "w") as f1:
                f1.write(doc +"\n")

        return

    @utz.time_decorator
    def embedding_split_to_faiss(self):
        """build and save faiss indes"""
        df_temp=pd.read_pickle(self.df_file_split)
        # convert to list
        embeds_list=[]
        len_embeds=0
        for _, row in df_temp.iterrows():
            embeds=row["embedding"]
            # assume equal length for all embeds???
            len_embeds=len(embeds)
            embeds_list.append(embeds)
        # convert to 2 dim numpy
        embeds_np_2dim=np.array(embeds_list)
        utz.print("embeds_np shape", embeds_np_2dim.shape)
        # build and load inded
        self.faiss_index = faiss.IndexFlatL2(len_embeds)  # L2 distance index
        self.faiss_index.add(embeds_np_2dim)
        utz.print("index stats", self.faiss_index.ntotal)
        faiss.write_index(self.faiss_index,  self.faiss_fn)
        return

    def embedding_split_load(self):
        self.df=pd.read_pickle(self.df_file_split)
        # self.faiss_index=faiss.read_index(self.faiss_fn)
        pass

    def copy_top_docs(self, n_prompts):
        """ copy to docs for ease of check"""
        for i  in range(n_prompts):
            with open(self.util_get_doc_filename(self.file_ids[i])) as f1:
                doc=f1.read()
            string_wiki=mwparserfromhell.parse(doc)
            string_text=string_wiki.strip_code()
            string_list=string_text.split(".")
            string_text="\n".join(string_list)
            dist=str(round(self.dists[i],3))
            with open("output/doc{}_{}_{}".format(i,self.file_ids[i] ,dist),"w")as f1:
                f1.write(string_text)
        return

    # search function
    @utz.time_decorator
    def get_top_docs(self,
        n_prompts :int,
        query_embedding,
    ):
        """create df with top N docs."""
        if  self.use_faiss:
            utz.print("using faiss")
            # 1 dim arrary
            query_embedding_np = np.array(query_embedding)
            # for faiss Reshape the array to 2 dim arrary  with a _single_ row equal to original 1 dim 
            query_embedding_np_2dim = query_embedding_np.reshape(1, -1)
            temp_dists, temp_ids = self.faiss_index.search(query_embedding_np_2dim, n_prompts)
            self.dists=temp_dists[0].tolist()
            self.file_ids=temp_ids[0].tolist()
        else:
            utz.print("using cosine as lambda")
            self.df1=self.df
            # add column
            self.df1["dist"]=self.df1.embedding.apply(lambda x : spatial.distance.cosine(x, query_embedding))
            # sort and take highest
            self.df1=self.df1.sort_values("dist",ascending=True).head(n_prompts)
            # copy  docs to disk
            self.dists=self.df1["dist"].tolist()
            self.file_ids=self.df1["file_id"].tolist()


        self.copy_top_docs(n_prompts)
        utz.print("dists",self.dists)
        utz.print("file_ids",self.file_ids)

        return

    def get_num_tokens(self,
                   text: str
                   ) -> int:
        """Return the number of tokens in a string."""
        encoding = tiktoken.encoding_for_model(test.GPT_MODEL)
        tokens=encoding.encode(text)
        num_tokens=len(tokens)
        return num_tokens

    def get_real_tokens_gpt2(self,text):
        # Load the GPT-2 tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokens = tokenizer.tokenize(text)
        utz.print(tokens)


    def prompt_build_from_docs(self,
                     query,
                     n_docs,
                     ):
        """Return a prompt for GPT, with relevant docs pulled from a dataframe."""
        prompt=""
        # add introduction
        introduction = 'Use the below articles on the 2022 Winter Olympics to answer the subsequent question. If the answer cannot be found in the articles, write "I could not find an answer."'
        introduction = ''
        prompt += introduction

        # add top N closest docs
        question = f"\n\nQuestion: {query}"
        for i in range(n_docs):
            with open(self.util_get_doc_filename(self.file_ids[i])) as f1:
                doc=f1.read()
                next_article = f'\n\nWikipedia article section:\n"""\n{doc}\n"""'
                num_tokens=self.get_num_tokens(prompt+ question + next_article)
                if (num_tokens> self.token_budget):
                    utz.print("budget exceeded :", num_tokens)
                    break
                else:
                    prompt += next_article

        # add user question
        prompt+=question
        return prompt

    @utz.time_decorator
    def ask(self,query,n_docs) :
        """Answers a query using GPT n_docs of    additonal docs."""
        if n_docs >0 :
            query_embedding_response = self.client.embeddings.create(
                model=test.EMBEDDING_MODEL,
                input=query,
            )
            query_embedding = query_embedding_response.data[0].embedding
            self.get_top_docs(n_docs, query_embedding)
            prompt = self.prompt_build_from_docs(query,n_docs)
        else:
            prompt = query


        messages = [
            {"role": "system", "content": "You answer questions about the 2022 Winter Olympics."},
            {"role": "user", "content": prompt},
        ]
        response = self.client.chat.completions.create(
            model=test.GPT_MODEL,
            messages=messages,
            temperature=0,
            seed=1,
            # temperature=1
        )
        answer=response.choices[0].message.content
        return answer

    def compare(self,query,max_n_prompts):
        """ compare experiments  with variable number of docs"""
        os.system("rm output/*")
        for n_prompts in range(0,max_n_prompts+1):
            answer=t1.ask(query,n_prompts)
            with open("output/answer_"+str(n_prompts) , "w" ) as file:
                file.write(answer+"\n")
            if n_prompts >0 :
                command = ("diff" + " output/answer_"+ str(n_prompts) +
                                    " output/answer_"+ str(n_prompts-1) +">"
                                    + " output/diff" + str(n_prompts))

                os.system(command)
        return



t1=test()

load_from_samples=False
if load_from_samples:
    t1.embedding_get_from_samples()
    t1.embedding_split()

# t1.use_faiss=False
t1.use_faiss=True
if t1.use_faiss:
    t1.embedding_split_to_faiss()
else:
    t1.embedding_split_load()

query = 'Which athletes won the gold medal in curling at the 2022 Winter Olympics?'

# single_query=True
single_query=False
if single_query:
    #  run single query
    answer=t1.ask(query,n_docs=1)
    utz.print(answer)
else:
    #  run repeatedly and compare results
    max_n_ndocs=2
    t1.compare(query,max_n_ndocs)



# query = answer + ' show only men'
# answer=t1.ask(query,n_docs=0)
# utz.print(answer)

# query = answer + ' now show only women'
# answer=t1.ask(query,n_docs=0)
# utz.print(answer)

# # t1.get_num_tokens("dear ,come here dear")
# # t1.get_real_tokens_gpt2("dear ,come here dear")
