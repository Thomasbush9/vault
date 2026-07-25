Ideas for experiments to run. 

### Increase Difficulty

Currently we have observed how with a simple scene the sender-receiver can communicate simply by according color or shape (not enough distractors). We should try to increase the difficulty and observe the results (consider whether to add the size). 


### Use pre-trained vision module 

At the moment we have only used custom trained vision modules -> should we move to a pretrained cnn? 

### Study over Input statistics 

Can we study how language grounds over the statistics of the input? 


### Study input representation changes during the communication game -> communication pact etc. 

We know that communication makes representation sender/receiver converge, while it makes sender/input different. Can we study it more and how it changes? Also observe to what the agents pay attention. 


### Dynamic game? 

Language's role is for social cooperation-> currently we run experiments for discrimination task, can we move to a more realistic task? 


There are few things that are worth add to the classic sender-receiver game setup: 

Human dialogues are embedded in a temporal context that defines the limits of the conversational space. Thus, it could be intersting to add a limit of steps (based also on the length of the message) for a conversation between agents. The second point is to treat the game a dialogue for mutual localization, so that both agents have the same distinct architecture (as they can be changed at each time and meet a different agent with a different perceptual history-> grounding the conceptual pact). Lastly, we should enforce a memory constraint (outside the loss function) that limits the vocab and the perceptual memory (here we can have two limits as in humans they operate differently). 

In this way we would have a final loss give by: distance + time? and a memory constraint. In this game the nice thing is that each agent can meet others with a different story but same architecture (although optimized differently for that timestep.)
