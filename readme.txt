The project is about automaton distillation. 
The code tests four different reinforcement learning methods: DQN without transfer learning, product MDP, policy distillation, and automaton distillation.

The overall organization:
	lib: the actual implementation of automaton transfer
	run: individual experiments. The files themselves should craft a config and call lib.main.run_training
		env: different environments that will be reused across different runs. Their configurations live here.
		utils: when two experiments require similar configurations, create a function in here to reduce repetition
		teacher: the teacher configs that serve either as DQN without transfer learning or as "teacher"
		target: the target configs that serve as the "student" in knowledge transfer

To recreate the training: 
First, install the required packages as shown in the requirement.txt file. 
Run one of the teacher configs. While it runs, it should create folders of checkpoints and logs. When running for the first time, it should print "NOT loading from the checkpoint"; otherwise, it will automatically load from the last checkpoint.
When the learning is done (when the steps reach maximum 1 million), it should create a file of automaton_q. This is extra information of the teacher that will be loaded later when running the student config.
Now run one of the corresponding target configs. Similarly, there will be checkpoints and logs.
All te logs are in tensorboard. To view, run tensorboard --logdir logs

The name of the teacher/target config corresponds to the learning methods, the experiment environment, and different award systems.  
Running the code requires a GPU, although the code can be modified to run without a GPU (using CPU instead of CUDA).
It takes long to run each config. For my computer with a RTX 3060, it takes about five hours to run each config, and the running time depends on what machine you are running it on.

