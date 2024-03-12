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

# utilities from py_utz
from utz import utz
# from utz import my_decorator
# from utzexc import UtzExc


class test:
    EMBEDDING_MODEL = "text-embedding-ada-002"
    # EMBEDDING_MODEL = "text-embedding-3-large"  # has too many dimensions compared to ada-002
    # EMBEDDING_MODEL = "text-embedding-3-small"  # same dimansions but distance reversed  compared to ada-002

    # GPT_MODEL = "gpt-3.5-turbo"
    # GPT_MODEL = "gpt-4"
    GPT_MODEL = "gpt-4-turbo-preview"

    def __init__(self) -> None:
        self.client = OpenAI()
        self.token_budget = 4096 - 500
        # self.token_budget =  10**5
        self.max_docs=3
        self.df_file="catalog/df.file"
        self.df_file_split="catalog/df_split.file"
        self.docs_dir = 'catalog/docs'
        pass



    def embedding_get(self):
        # download pre-chunked text and pre-computed embeddings
        # this file is ~200 MB, so may take a minute depending on your connection speed
        """get list of embedding for wiki from external site and save it"""
        print(datetime.now(),"starting to read remote csv")
        embeddings_path = "https://cdn.openai.com/API/examples/data/winter_olympics_2022.csv"
        self.df = pd.read_csv(embeddings_path)

        print(datetime.now(),"starting to convert  csv")
        # convert embeddings from CSV str type back to list type
        self.df['embedding'] = self.df['embedding'].apply(ast.literal_eval)
        print("len of embedding df ", len(self.df))

        # print(datetime.now(),"starting save to file csv")
        # self.df.to_csv(self.df_file, index=False)
        print(datetime.now(),"starting  save to file pickle")
        self.df.to_pickle(self.df_file)
        print(datetime.now(),"finished all ")
        pass

    def util_get_doc_filename(self,file_id):
        return self.docs_dir +"/" +str(file_id).zfill(5)

    @utz.time_decorator
    def embedding_split(self):
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
    def embedding_to_faiss(self):
        self.df=pd.read_pickle(self.df_file_split)
        # convert to numpy
        embeds_list=[]
        len_embeds=0
        for irow, row in self.df.iterrows():
            embeds=row["embedding"]
            len_embeds=len(embeds)
            embeds_list.append(embeds)
        self.np=np.array(embeds_list)
        # build and load inded
        self.index = faiss.IndexFlatL2(len_embeds)  # L2 distance index
        self.index.add(self.np)
        print("index stats", self.index.ntotal)
        faiss.write_index(self.index, "catalog/faiss.index")
        return

    def embedding_load(self):
        self.df=pd.read_pickle(self.df_file_split)
        pass

    # search function
    @utz.time_decorator
    def get_top_docs(self,
        n_prompts :int,
        query_embedding,
    ):
        """create df with top N docs."""

        self.df1=self.df
        # add column
        self.df1["dist"]=self.df1.embedding.apply(lambda x :1 - spatial.distance.cosine(x, query_embedding))
        # sort and take highest
        self.df1=self.df1.sort_values("dist",ascending=False).head(n_prompts)
        # random docs  ???
        # self.df1=self.df1.sample(n=n_prompts,random_state=42)

        # copy  docs to disk
        import mwparserfromhell
        dists=self.df1["dist"].tolist()
        file_ids=self.df1["file_id"].tolist()
        for i  in range(n_prompts):
            with open(self.util_get_doc_filename(file_ids[i])) as f1:
                doc=f1.read()
            string_wiki=mwparserfromhell.parse(doc)
            string_text=string_wiki.strip_code()
            string_list=string_text.split(".")
            string_text="\n".join(string_list)
            dist=str(round(dists[i],3))
            with open("output/doc{}_{}_{}".format(i,file_ids[i] ,dist),"w")as f1:
                f1.write(string_text)
        return

    def get_num_tokens(self,
                   text: str
                   ) -> int:
        """Return the number of tokens in a string."""
        encoding = tiktoken.encoding_for_model(test.GPT_MODEL)
        tokens=encoding.encode(text)
        num_tokens=len(tokens)
        # print("====num_tokens", num_tokens)
        # print(tokens)
        return num_tokens

    def get_real_tokens_gpt2(self,text):
        # Load the GPT-2 tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokens = tokenizer.tokenize(text)
        print(tokens)


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
        file_ids=self.df1["file_id"].tolist()
        for i in range(n_docs):
            with open(self.util_get_doc_filename(file_ids[i])) as f1:
                doc=f1.read()
                next_article = f'\n\nWikipedia article section:\n"""\n{doc}\n"""'
                num_tokens=self.get_num_tokens(prompt+ question + next_article)
                if (num_tokens> self.token_budget):
                    print("budget exceeded :", num_tokens)
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

    def comp(self,query):
        """ compare experiments  with variable number of docs"""
        os.system("rm output/*")
        for n_prompts in range(0,self.max_docs+1):
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

# t1.embedding_get()
# t1.embedding_split()

# t1.embedding_split()
t1.embedding_to_faiss()
exit()


t1.embedding_load()
query = 'Which athletes won the gold medal in curling at the 2022 Winter Olympics?'
t1.comp(query)
# print(t1.ask(query,n_docs=1))





# # t1.get_num_tokens("dear ,come here dear")
# # t1.get_real_tokens_gpt2("dear ,come here dear")
