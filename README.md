# TODO: 
0, design the structure of the robot, decide the scenario \
1, add slam and nav2, to enable the moving function of lekiwi \
2, using other policies \
3, multiple gpu training \
4, 


# robot_learning
robot_learning


python3 -m lerobot.scripts.eval --policy.path=lerobot/diffusion_pusht --env.type=pusht --eval.batch_size=10 --eval.n_episodes=10 --policy.use_amp=false --policy.device=cpu
