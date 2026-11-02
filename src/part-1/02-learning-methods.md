Supervised learning, reinforcement learning, self-supervised learning, etc 
Automatic labelling + supervised learning - A model or sensor-based process generates labels; another model is trained using those labels. 
Self-supervised learning - The training objective is created from the data itself, for example by predicting masked image regions or future observations. Manual object labels are not necessary for that particular task.
Unsupervised learning - The algorithm learns structure or representations without conventional ground-truth labels.
Reinforcement learning - A policy learns using rewards or other feedback from interactions or simulated environments.


How would you check an unlabeled dataset in practice? 
Data integrity - Automated checks for corrupted files, missing frames, duplicate sequences, timestamp inconsistencies, invalid image dimensions and incorrect metadata. Versioning and data provenance should also be documented.

Learning strategy                 Tesla              NVIDIA Alpamayo                       Mobileye 
Supervised learning / imitation  Yes                  Yes, through supervised fine-tuning  Yes, particularly in perception
Automatic label generation         Yes               Yes                                    Yes  
Self-supervised / unlabeled learning   Not the best description of the whole public pipeline    Possible within foundation-model pretraining; not the full Alpamayo training strategy     Not the best description of the whole public pipeline 
Reinforcement learning             Yes                 Yes                                  Yes, for driving-policy training through ACI #
Learning from simulated driving     Used within its development ecosystem Yes, including closed-loop RL with AlpaGym    Yes, including self-play simulation for ACI 
