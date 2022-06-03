# AI506

### Make embedding (svd)
```
python embedding.py --length=512
```

### Run completion task
``` 
python3 completion.py embedding_mlp --embedding=Word2vec128 --inputsize=128 --hiddensize=512 --filename=completion_answer
``` 
