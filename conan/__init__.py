import gym
gym.register(
    id='conan_nav-v0',
    entry_point='conan.training.env:Nav_Env',
    max_episode_steps=10000,
    kwargs={'reward': True,
            'length': 5000,
            'seed': None,
            'boss': True,
            'recover': False,
            'footprints': False,
            'view_type': 'symbolic'})
gym.register(
    id='conan_nav-v1',
    entry_point='conan.gen:Nav_Env',
    max_episode_steps=10000,
    kwargs={'reward': True,
            'length': 10000,
            'seed': None,
            'boss': True,
            'recover': False,
            'footprints': False,
            'view_type': 'visual'})
