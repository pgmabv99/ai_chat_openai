# action log + todo
-function call from model
-embedding
---curlings  in 2022 games
--- N promts selected
--- without prompts results men= swiss, women = swiss
--- with N prompts results  men = canada, women= swiss
--- CORRECT  from website   men=sweden , women =Great britain

- study compare as n_docs 1 --> 5
- switch to compatible models
- refactor

- TODO use vector DB
-

# profiling
python3 -m cProfile -o output/profile.txt t3_embedding_search.py
snakeviz output/profile.txt

# performance
- read remote CSV - 30sec
- convert to binary - 24
- convert to textfile local - 7 sec
- convert to pickle file local - 0.5 sec
- resort/filter  df - 0.5 sec vs vector ???
```
2024-03-08 20:49:14.246253 starting to read remote csv
2024-03-08 20:49:44.242880 starting to convert  csv
len of embedding df  6059
2024-03-08 20:50:08.115247 starting save to file csv
2024-03-08 20:50:15.366799 starting  save to file pickle
2024-03-08 20:50:15.890128 finished all
```
- lremoving large pd columns with text does not affect sort speed
```
    with large column in sort 2.293	0.7642	t3_embedding_search.py:84(get_top_docs)
    without largecolu 2.35	0.7832	t3_embedding_search.py:84(get_top_docs)
```

# accuracy (seed=1 used)
- question "Which athletes won the gold medal in curling at the 2022 Winter Olympics?
- 0 doc -  Basic info :   just teams listed correctly with recent model  gpt-4-turbo-preview
- adding 1 doc   add details such as team members
- more  docs   does not change the answer, just different synonym (consisting/including) and different placement of the countryname

# model compatability (gotcha)
- cannot compare embdedding from different models
    ```
    EMBEDDING_MODEL = "text-embedding-ada-002"
    # EMBEDDING_MODEL = "text-embedding-3-large"  # has too many dimensions compared to ada-002
    # EMBEDDING_MODEL = "text-embedding-3-small"  # same dimansions but distance reversed  compared to ada-002
    ```

# references
https://cookbook.openai.com/examples/question_answering_using_embeddings