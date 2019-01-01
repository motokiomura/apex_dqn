# Ape-X DQN
This is TensorFlow implementation of **Ape-X DQN** : [DISTRIBUTED PRIORITIZED EXPERIENCE REPLAY](https://openreview.net/pdf?id=H1Dy---0Z)


In my code, 
`multiprocessing` is adopted to implement distribution of a learner and actors and
`multiprocessing.Queue` is used to pass network parameters and trajectories among processes.

Environment : 
[OpenAI Gym Atari 2600 games](https://gym.openai.com/envs/#atari)


## Usage
```
$ python apex_dqn.py --num_actors 3 --env_name Alien-v0 --replay_memory_size 200000
```

All arguments are described in `apex_dqn.py`


## Results

After 10,000 episodes (8 actors)

![alien](https://user-images.githubusercontent.com/39490801/49695191-9cfbf380-fbda-11e8-879d-35bc819deb4c.gif)

### Learning curves

![alien_result](https://user-images.githubusercontent.com/39490801/49695268-f1ec3980-fbdb-11e8-979d-ed0307eb79a0.png)
