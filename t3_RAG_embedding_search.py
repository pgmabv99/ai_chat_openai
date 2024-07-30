""" test OpenAI building of prompts using proximity search on related documents """
# imports
import ast  # for converting embeddings saved as strings back to arrays
from openai import OpenAI # for calling the OpenAI API

import tiktoken  # for counting tokens

# alternatives for vector search
from scipy import spatial  # by hand compute cosine distance
import faiss   # use fascebook search package
#  ??
from transformers import GPT2Tokenizer

# from datetime import  datetime
import os
import shutil
import mwparserfromhell  # to parse wiki pages

import pandas as pd   #for direct cosine distrane
import numpy as np    #for faiss

# utilities from py_utz
from utz import utz
from datetime import datetime

class search:


    def __init__(self) -> None:
        utz.logset()
        self.openai_cln = OpenAI()
        self.token_budget = 4096 - 500
        # self.token_budget =  10**5
        self.df_file="catalog/df.file"
        self.df_file_split="catalog/df_split.file"
        self.docs_dir = 'catalog/docs'
        self.faiss_fn="catalog/faiss.index"
        self.query_model="gpt-4-turbo-preview"
        self.query_model="gpt-4o-mini"
        # self.query_model="gpt-4o"
        self.embedding_model="text-embedding-ada-002"
        utz.print("using query  model{} embed model{}".format(self.query_model, self.embedding_model))
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
    def embedding_pickle_to_faiss(self):
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

    def embedding_pickle_load(self):
        self.df=pd.read_pickle(self.df_file_split)
        pass

    def copy_top_docs(self):
        """ copy to docs for ease of checking"""
        for i,file_id  in enumerate(self.file_ids):
            with open(self.util_get_doc_filename(file_id)) as f1:
                doc=f1.read()
            string_wiki=mwparserfromhell.parse(doc)
            string_text=string_wiki.strip_code()
            string_list=string_text.split(".")
            string_text="\n".join(string_list)
            dist=str(round(self.dists[i],3))
            with open("output/doc{}_{}_{}".format(i,file_id ,dist),"w")as f1:
                f1.write(string_text)
        return

    # 
    @utz.time_decorator
    def get_top_docs(self,
                        n_docs :int,
                        query,
                    ):
        """create a list  with top N docs closest to query """
        
        # get embedding for the current query
        query_embedding_response = self.openai_cln.embeddings.create(
            model=self.embedding_model,
            input=query,
        )
        query_embedding = query_embedding_response.data[0].embedding
        
        if  self.use_faiss:
            utz.print("using faiss")
            # 1 dim arrary
            query_embedding_np = np.array(query_embedding)
            # for faiss Reshape the array to 2 dim arrary  with a _single_ row equal to original 1 dim 
            query_embedding_np_2dim = query_embedding_np.reshape(1, -1)
            # do search
            temp_dists, temp_ids = self.faiss_index.search(query_embedding_np_2dim, n_docs)
            # make a list of top N
            self.dists=temp_dists[0].tolist()
            self.file_ids=temp_ids[0].tolist()
        else:
            utz.print("using cosine as lambda")
            self.df1=self.df
            # add column
            self.df1["dist"]=self.df1.embedding.apply(lambda x : spatial.distance.cosine(x, query_embedding))
            # sort and take highest
            self.df1=self.df1.sort_values("dist",ascending=True).head(n_docs)
            # make a list of top N
            self.dists=self.df1["dist"].tolist()
            self.file_ids=self.df1["file_id"].tolist()


        self.copy_top_docs()
        utz.print("dists",self.dists)
        utz.print("file_ids",self.file_ids)

        return


    def get_real_tokens_gpt2(self,text):
        # Load the GPT-2 tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokens = tokenizer.tokenize(text)
        utz.print(tokens)


    def prompt_build_from_docs(self,
                     query,
                     ):
        """Return a prompt for GPT, with addition documents."""
        prompt=""
        # add introduction
        introduction = 'Use the below articles on the 2022 Winter Olympics to answer the subsequent question. If the answer cannot be found in the articles, write "I could not find an answer."'
        prompt += introduction

        # add query 
        query_plus = f"\n\nQuestion: {query}"
        
        # add top N closest docs from the list
        encoding = tiktoken.encoding_for_model(self.embedding_model)
        for i, file_id in enumerate(self.file_ids):
            with open(self.util_get_doc_filename(file_id)) as f1:
                doc=f1.read()
                doc_plus = f'\n\nWikipedia article section:\n"""\n{doc}\n"""'
                # get number of tokens
                tokens=encoding.encode(prompt+ query_plus + doc_plus)
                num_tokens=len(tokens)
                utz.print("== doc {} len in bytes {} num_token {}".format(i, len(doc_plus),num_tokens))
                if (num_tokens> self.token_budget):
                    utz.print("budget exceeded :", num_tokens)
                    break
                else:
                    prompt += doc_plus

        # add user question (again?)
        prompt+=query_plus
        
        return prompt

    @utz.time_decorator
    def ask(self,query,n_docs) :
        """Answers a query using GPT n_docs of    additonal docs."""
        
        if n_docs >0 :

            # get documents that are close to query
            self.get_top_docs(n_docs, query)
            # add docs to prompt
            prompt = self.prompt_build_from_docs(query)
        else:
            #  just use the query 
            prompt = query

        #  build messages and get response content
        messages = [
            {"role": "system", "content": "You answer questions about the 2022 Winter Olympics."},
            {"role": "user", "content": prompt},
        ]
        response = self.openai_cln.chat.completions.create(
            model=self.query_model,
            messages=messages,
            temperature=0,
            seed=1,
            # temperature=1
        )
        content=response.choices[0].message.content
        
        return content

    def compare(self,query,max_n_prompts):
        """ compare experiments  with variable number of docs"""

        for n_prompts in range(0,max_n_prompts+1):
            answer=self.ask(query,n_prompts)
            with open("output/answer_"+str(n_prompts) , "w" ) as file:
                file.write(answer+"\n")
            if n_prompts >0 :
                command = ("diff " + "output/answer_"+ str(n_prompts) + " "
                                   + "output/answer_"+ str(n_prompts-1) +" >"
                                   + "output/diff" + str(n_prompts))

                os.system(command)
        return
    
    def output_init(self):
        # Re-create the output folder
        if os.path.exists("output"):
            shutil.rmtree("output")
        os.makedirs("output")
        
        
    #  test runs
    def run_query(self,query):
        self.output_init()

        load_from_samples=False
        if load_from_samples:
            # build pickle
            self.embedding_get_from_samples()
            self.embedding_split()

        self.use_faiss=False
        # self.use_faiss=True
        if self.use_faiss:
            self.embedding_pickle_to_faiss()
        else:
            self.embedding_pickle_load()



        single_query=True
        # single_query=False
        if single_query:
            #  run single query
            n_docs=2
            # n_docs=0
            answer=self.ask(query,n_docs)
            utz.print(answer)
            ct = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            # with open("output_old/answer"+ct,"w") as f1:
            with open("output_old/answer"+self.query_model+"_ndoc_"+str(n_docs)+"_"+ct,"w") as f1:
                f1.write(answer)
            return answer
        else:
            #  run repeatedly and compare results
            max_n_ndocs=2
            self.compare(query,max_n_ndocs)
            return "comparison done. see output "

    def run_decode_voice(self):
        audio_file= open("catalog/untitled.wav", "rb")
        transcription =self.openai_cln.audio.transcriptions.create(
                        model="whisper-1", 
                        file=audio_file
                        )
        utz.print(transcription.text)



if __name__ == "__main__":
    s1=search()
    query = 'Which athletes won the gold medal in curling at the 2022 Winter Olympics?'
    query += '.format as table'
    s1.run_query(query)

    # s1.run_decode_voice()




