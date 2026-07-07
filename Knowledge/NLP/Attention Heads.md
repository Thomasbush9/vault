Notes from: [https://lena-voita.github.io/posts/acl19_heads.html]{Attention Heads}

**Heads Importance**: a confident head assigns a high proportion of attention to a single token. Here confidence can be measured by the average of its max attention weight (without EOS)-> there are few confident heads

**Layerwise Relevance Propagation (LRP):** it allows to measure heads' importance. 
