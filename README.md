# Super-Mario-RL

**GOAL**: I wanted to replicate the success of using Reinforcement Learning for Super Mario Bros and understand basic RL concepts over the span of 4-5 weeks. I then lead a team where I tried to teach these concepts and help my peers to have a better understanding of RL. I followed this video [here](https://www.youtube.com/watch?v=_gmQZToTMac&t=1310s&ab_channel=SourishKundu). This project is purely for educational purposes and code belongs to the original owner. The code was dissected and rewritten in order to understand the workings of RL.

## Installation

**First, create a virtual environment**

You can use whatever Python version works for you. 

```bash
conda create --name <name of env> python=3.10.12 
```

Then, activate the environment.

```bash
conda activate <name of env>
```

**Then, install PyTorch v2.2.1**

Follow the instructions on [PyTorch's website](https://pytorch.org/get-started/locally/).

Using pip, the command is:

```bash
pip3 install torch torchvision torchaudio
```

**Finally, install the rest of the requirements**

```bash
pip install -r requirements.txt
```

## Results
After training the model for 50000 iterations over the span of 3+ days using a 3070 GPU running locally, we were able to successully replicate the result of Mario completing World 1-1. Although this wasn't 100% on the test runs, the agent improved drastically over time compared to the random movements initially. We then did a live test run to a class of 30+ students, which Mario completed succesfully.

## Learning Outcomes
From this project, I learned about the basics of Reinforcement Learning, conceptually and using the OpenAI GYM API. I gave lessons on the concepts of agents, rewards, bellman-optimality, etc... Overall, this was a techincal challenge to learn something and teach it without having a large amount of knowledge on the topic but I was proud that I was able to answer questions my team had. 

## Resources
- OpenAI Gym Documentation: (https://gymnasium.farama.org/)
- Deep Q-Network Explained: (https://www.youtube.com/watch?v=x83WmvbRa2I&ab_channel=CodeEmporium)
- Deep RL with OpenAIGym: (https://www.youtube.com/watch?v=YLa_KkehvGw&t=181s&ab_channel=NeuralNine)
- PyTorch Super Mario Tutorial: (https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html)
- PyTorch Replay Buffer: (https://pytorch.org/rl/reference/data.html)
- CNN Explained: (https://poloclub.github.io/cnn-explainer/)
  
