    if training_args.log_to_wandb:
        wandb.login(key="569ce0cbd5d7c96161dc36a682a88a18bd3f27dd")
        # group_name = training_args.job_name + "_" + data_args.task_name
        wandb.init(
            name=training_args.job_name,
            # group=group_name,
            project="not-decision-transformer",
            config=vars(training_args),
        )
        # wandb.watch(model)  # wandb has some bug