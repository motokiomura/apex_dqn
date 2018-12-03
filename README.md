# apex_dqn
This is TensorFlow implementation of [**Ape-X DQN** : DISTRIBUTED PRIORITIZED EXPERIENCE REPLAY](https://openreview.net/pdf?id=H1Dy---0Z)

Environment : 
[OpenAI Gym Atari 2600 games](https://gym.openai.com/envs/#atari)


## Usage
```
$ python apex_dqn.py --num_actors 3 --env_name Alien-v0 --replay_memory_size 200000
```

All arguments are described in `apex_dqn.py`
