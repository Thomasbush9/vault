It could be intersteting to:
1. Train a small model on the tiny stories dataset + report the paper replication etc. 
2. Apply some sort of fine tuning + LoRA 

It would be also interesting to check whether we could fine tune two models with LoRA and deterministically switch the matrices of low ranks weights or mix them to change the behaviour of the model. For example we could: train one model on a semantic task, another one something else and then see the performances against a baseline trained on the union. 