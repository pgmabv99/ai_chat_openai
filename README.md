-function call from model
-embedding
---curlings  in 2022 games
--- N promts selected
--- without prompts results men= swiss, women = swiss
--- with N prompts results  men = canada, women= swiss
--- CORRECT  from website   men=sweden , women =Great britain

--profiling
python3 -m cProfile -o output/profile.txt t3_embedding_search.py
snakeviz output/profile.txt