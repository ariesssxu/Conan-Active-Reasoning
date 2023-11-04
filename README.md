![Demo](./figs/diamond.gif)

# Active Reasoning in an Open-World Environment  

Manjie Xu, Guangyuan Jiang, Wei Liang, Chi Zhang, Yixin Zhu  

Thirty-Seventh Annual Conference on Neural Information Processing Systems (NeurIPS 2023)  

<a href='https://yzhu.io/publication/intent2023neurips/paper.pdf'>
  <img src='https://img.shields.io/badge/Paper-CoRe-green?style=plastic&logo=arXiv&logoColor=green' alt='Paper CoRe'>
</a>
</a>
<a href='https://sites.google.com/view/conan-active-reasoning'>
  <img src='https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
</a>
<a href='https://vimeo.com/878540519'>
  <img src='https://img.shields.io/badge/Project-Demo-red?style=plastic&logo=Youtube&logoColor=red' alt='Project Page'>
</a>

## Brief Introduction
Recent advances in vision-language learning have achieved notable success on complete-information question-answering datasets through the integration of extensive world knowledge. Yet, most models operate passively, responding to questions based on pre-stored knowledge. In stark contrast, humans possess the ability to actively explore, accumulate, and reason using both newfound and existing information to tackle incomplete-information questions.   

In response to this gap, we introduce 🔍Conan, an interactive open-world environment devised for the assessment of active reasoning. 🔍Conan facilitates active exploration and promotes multi-round abductive inference, reminiscent of rich, open-world settings like Minecraft. Diverging from previous works that lean primarily on single-round deduction via instruction following,
Conan compels agents to actively interact with their surroundings, amalgamating new evidence with prior knowledge to elucidate events from incomplete observations. 

<div align=center>
  <img src=./figs/intro.png />
</div>

<!-- Our analysis on Conan underscores
the shortcomings of contemporary state-of-the-art models in active exploration and
understanding complex scenarios. Additionally, we explore Abduction from Deduction, where agents harness Bayesian rules to recast the challenge of abduction as a deductive process. Through Conan, we aim to galvanize advancements in active reasoning and set the stage for the next generation of AI agents adept at dynamically engaging in environments. -->

## 🔍TODO ![coverage](https://img.shields.io/badge/coverage-60%25-yellowgreen) ![version](https://img.shields.io/badge/version-1.0.0-purple)

- [√] 🔍Conan environment code.
- [√] 🔍Conan baselines code.
- [√] 🔍Conan question generation (for new questions).
- [ ] 🔍Conan web version.
- [ ] 🔍Conan checkpoints.
## 🔍Code Structure
```
├── Conan
│     ├── analysis     # Code for analysis in our paper 
│     ├── playground      # Core components of the Conan playground 
│     ├── gen          # Code replated to the trace generation and question generation
│     ├── reasoning    # Visual-language models and training/reasoning code
│     ├── __init__.py
│     └── run_gui.py
├── figs               # figs for demostration
├── LICENCE
├── README.md
└── setup.py
```
## 🔍Install
We recommend using Conda to install 🔍Conan.
```
conda create -n conan python=3.10    # create a new conda environment
conda activate conan                 # activate the environment
cd conan && pip install -e .         # install dependencies
```
## 🔍Conan Playground

Originating from the [Crafter](https://github.com/danijar/crafter), 🔍Conan offers an extensive assortment of interactive items, tasks and traces in the new playground.

To play as an agent in the playground with GUI, run:
```
python playground/run_gui.py --record all --length 500 --recover False --boss True --footprints True
```
Some common parameters:
- ``--record`` controls the record level. ``all`` records all the data, including the environments, steps and the video. See ``playground/recorders`` for more details. 
- ``--recover`` controls whether to recover the previous tragectory. If ``True``, the previous data will be recovered. In the ``env.py``, we automatically recover from the npz file in the default log path.
- ``--boss`` controls whether to play the game as a boss. In this mode, you will not be hurt in the game. This mode is useful when simulating the detective.
- ``--footprints`` controls whether your movement will leave footprints on the ground. Set it as ``True`` if you want to record footprint traces in the environment.
More parameters can be found in ``playground/run_gui.py``.

## 🔍Generation
### Traces
The above command will generate a trace file in the default log path. Actually, this is what the vandal does in the paper. To generate more traces automatically, run:
```
python gen/task/gen.py --length 300 --num 100 --boss True --other_agents True --render True --save_path save/tmp
```
This command will generate 100 traces with 300 steps each. The traces will be saved in the ``save/tmp`` folder. The ``--render`` parameter controls whether to render the trace. 

Some other useful parameters:
- ``--other_agents`` parameter controls whether to include other agents in the environment. 
- ``--task`` parameter controls the task type. Default is ``none``, which means all the tasks. You can specify the task type you want to generate. See ``gen/task/tasks.yaml`` for more details.

### Questions ![coverage](https://img.shields.io/badge/in-development-pink)
Questions in 🔍Conan fall into three primary categories: Intent (local intent), Goal (global goal), and Survival (agent’s survival status change). To generate questions from saved traces, run:
```
TBD
```

## 🔍Explorer
🔍Conan casts the abductive reasoning challenge as a detective game, necessitating a detective to
efficiently explore and gather information from the environment to deduce plausible explanations. To train an explorer, run:
```
python training/exploration_pretrain/run_{MODEL}.py
```
where ``{MODEL}`` can be DQN, TRPO and RecurrentPPO. The training code is based on Stable Baselines 3.

## 🔍Vision-Language Reasoner
We employ a multi-choice question-answering paradigm. We evaluate several well-established multimodal models as the reasoner in 🔍Conan. To adopt a reasoner, run:
```
cd training/vl/{MODEL} && bash train.sh
```
where ``{MODEL}`` can be clip_trans, flamingo_mini and frozernbilm.

## 🔍Citation
If you find the paper and/or the code helpful, please cite us.
```
@inproceedings{xu2023active,
  title={Active Reasoning in an Open-World Environment},
  author={Xu, Manjie and Jiang, Guangyuan and Liang, Wei and Zhang, Chi and Zhu, Yixin},
  booktitle={NeurIPS},
  year={2023}
}
```