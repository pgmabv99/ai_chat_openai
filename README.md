# action log + todo
-function call from model
-embedding
---curlings  in 2022 games
--- N promts selected
--- without prompts results men= swiss, women = swiss
--- with N prompts results  men = canada, women= swiss
--- CORRECT  from website   men=sweden , women =Great britain

- study compare as n_docs 1 --> 5
- to use vector DB

# profiling
python3 -m cProfile -o output/profile.txt t3_embedding_search.py
snakeviz output/profile.txt

# performance
- read remote CSV - 30sec
- convert to binary - 24
- convert to textfile local - 7 sec
- convert to pickle file local - 0.5 sec
```
2024-03-08 20:49:14.246253 starting to read remote csv
2024-03-08 20:49:44.242880 starting to convert  csv
len of embedding df  6059
2024-03-08 20:50:08.115247 starting save to file csv
2024-03-08 20:50:15.366799 starting  save to file pickle
2024-03-08 20:50:15.890128 finished all
```

# accuracy (seed=1 used)
- adding 1 doc dds details
- after 3 docs adding more does not change the answe